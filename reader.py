"""
reader.py — Kafka consumer for the 'publish-event' topic.

Reads messages every 10 seconds (via a scheduled polling thread), deserialises
the JSON payload into typed Python dataclasses that mirror the Java entity model,
and prints a structured summary to stdout.

Schema notes (from Java entities):
  - OrderedItems.orderInitiatedDate and orderUpdatedDate are @JsonIgnore
    → they are NOT present in the Kafka payload and are excluded from the model.
  - OrderedItems.orderItemAuditList is NOT @JsonIgnore
    → it IS present in the payload and is fully mapped.
  - OrderItemAudit.orderedItems and .items are @JsonIgnore → excluded.

Dependencies:
    pip install kafka-python-ng

Environment variables (or edit the KafkaConfig dataclass below):
    KAFKA_BOOTSTRAP_SERVERS   default: localhost:9092
    KAFKA_TOPIC               default: publish-event
    KAFKA_GROUP_ID            default: order-reader-group
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import threading
from dataclasses import dataclass, field
from decimal import Decimal
from typing import List, Optional

from kafka import KafkaConsumer
from kafka.errors import KafkaError

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("order-reader")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KafkaConfig:
    bootstrap_servers: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    topic: str             = os.getenv("KAFKA_TOPIC", "order-publish")
    group_id: str          = os.getenv("KAFKA_GROUP_ID", "order-reader-group")
    poll_interval_seconds: int = 10
    poll_timeout_ms: int       = 5_000   # max wait inside each poll call
    auto_offset_reset: str     = "earliest"
    enable_auto_commit: bool   = True


# ---------------------------------------------------------------------------
# Domain model
#
# Only fields that Jackson will serialise into the Kafka payload are modelled.
# Fields annotated @JsonIgnore in Java (orderInitiatedDate, orderUpdatedDate,
# orderDetails, orderedItems, items) are intentionally absent — they will
# never appear in the message and mapping them would cause KeyErrors.
# ---------------------------------------------------------------------------

@dataclass
class OrderItemAudit:
    """
    Mirrors OrderItemAudit (excluding @JsonIgnore relations orderedItems / items).
    Jackson serialises boolean getters as camelCase without the 'is' prefix
    when using default settings, but Spring Boot commonly retains 'is' — both
    variants are handled in from_dict below.
    """
    order_tracer_code: int  = 0
    is_active: bool         = False
    is_packed: bool         = False
    is_delivered: bool      = False
    quantity: int           = 0
    size: str               = ""
    item_price: float       = 0.0

    @classmethod
    def from_dict(cls, data: dict) -> "OrderItemAudit":
        """
        Maps a dict from the JSON payload to an OrderItemAudit instance.
        Jackson may serialise boolean fields as 'active'/'isActive' depending
        on the getter name, so both keys are checked.
        """
        def _bool(key_with_is: str) -> bool:
            # Try 'isActive' first, fall back to 'active'
            key_without = key_with_is[2].lower() + key_with_is[3:]
            return bool(data.get(key_with_is, data.get(key_without, False)))

        return cls(
            order_tracer_code=int(data.get("orderTracerCode", 0)),
            is_active=_bool("isActive"),
            is_packed=_bool("isPacked"),
            is_delivered=_bool("isDelivered"),
            quantity=int(data.get("quantity", 0)),
            size=str(data.get("size", "")),
            item_price=float(data.get("itemPrice", 0.0)),
        )

    def __str__(self) -> str:
        status_flags = (
            f"active={self.is_active}, "
            f"packed={self.is_packed}, "
            f"delivered={self.is_delivered}"
        )
        return (
            f"    OrderItemAudit(\n"
            f"      orderTracerCode = {self.order_tracer_code}\n"
            f"      quantity        = {self.quantity}\n"
            f"      size            = {self.size!r}\n"
            f"      itemPrice       = {self.item_price:.2f}\n"
            f"      status          = [{status_flags}]\n"
            f"    )"
        )


@dataclass
class OrderedItems:
    """
    Mirrors OrderedItems.
    Excluded (@JsonIgnore): orderInitiatedDate, orderUpdatedDate, orderDetails.
    Included: oId, mainActive, processed, totalPrice, paidPrice,
              orderItemAuditList (now serialised since @JsonIgnore was removed).
    """
    o_id: int                                   = 0
    main_active: bool                           = False
    processed: bool                             = False
    total_price: Decimal                        = Decimal("0.00")
    paid_price: Decimal                         = Decimal("0.00")
    order_item_audit_list: List[OrderItemAudit] = field(default_factory=list)

    # ----- factory / object-mapper equivalent -----

    @classmethod
    def from_dict(cls, data: dict) -> "OrderedItems":
        """
        Deserialises a raw dict into an OrderedItems instance, recursively
        mapping the nested orderItemAuditList.
        Equivalent to Jackson's ObjectMapper.readValue(json, OrderedItems.class).
        """
        audit_raw  = data.get("orderItemAuditList") or []
        audit_list = [OrderItemAudit.from_dict(a) for a in audit_raw]

        return cls(
            o_id=int(data.get("oId", 0)),
            main_active=bool(data.get("mainActive", False)),
            processed=bool(data.get("processed", False)),
            total_price=Decimal(str(data.get("totalPrice", "0.00"))),
            paid_price=Decimal(str(data.get("paidPrice", "0.00"))),
            order_item_audit_list=audit_list,
        )

    @classmethod
    def from_json(cls, raw_json: str) -> "OrderedItems":
        """Entry point — accepts the raw JSON string from Kafka."""
        data = json.loads(raw_json)
        return cls.from_dict(data)

    def __str__(self) -> str:
        audit_lines = (
            "\n".join(str(a) for a in self.order_item_audit_list)
            if self.order_item_audit_list
            else "    (none)"
        )
        return (
            f"OrderedItems(\n"
            f"  oId          = {self.o_id}\n"
            f"  mainActive   = {self.main_active}\n"
            f"  processed    = {self.processed}\n"
            f"  totalPrice   = {self.total_price}\n"
            f"  paidPrice    = {self.paid_price}\n"
            f"  auditItems   =\n{audit_lines}\n"
            f")"
        )


# ---------------------------------------------------------------------------
# Kafka consumer
# ---------------------------------------------------------------------------

class OrderKafkaReader:
    """
    Polls the Kafka topic on a background daemon thread every
    `config.poll_interval_seconds` seconds and maps each message to an
    OrderedItems instance.
    """

    def __init__(self, config: KafkaConfig) -> None:
        self._config   = config
        self._stop_evt = threading.Event()
        self._thread   = threading.Thread(
            target=self._run,
            name="kafka-reader",
            daemon=True,
        )

    # ----- public API -----

    def start(self) -> None:
        log.info(
            "Starting Kafka reader — topic=%s, brokers=%s, interval=%ss",
            self._config.topic,
            self._config.bootstrap_servers,
            self._config.poll_interval_seconds,
        )
        self._thread.start()

    def stop(self) -> None:
        log.info("Shutdown requested — signalling reader thread …")
        self._stop_evt.set()
        self._thread.join(timeout=15)
        log.info("Reader thread stopped.")

    # ----- internal -----

    def _build_consumer(self) -> KafkaConsumer:
        return KafkaConsumer(
            self._config.topic,
            bootstrap_servers=self._config.bootstrap_servers,
            group_id=self._config.group_id,
            auto_offset_reset=self._config.auto_offset_reset,
            enable_auto_commit=self._config.enable_auto_commit,
            value_deserializer=lambda b: b.decode("utf-8"),
            key_deserializer=lambda b: b.decode("utf-8") if b else None,
            consumer_timeout_ms=self._config.poll_timeout_ms,
        )

    def _run(self) -> None:
        consumer: Optional[KafkaConsumer] = None
        try:
            consumer = self._build_consumer()
            log.info("Consumer connected. Polling every %ds …", self._config.poll_interval_seconds)

            while not self._stop_evt.is_set():
                self._poll_once(consumer)
                # Sleep in small increments so stop_evt is checked promptly
                self._stop_evt.wait(timeout=self._config.poll_interval_seconds)

        except KafkaError as exc:
            log.error("Fatal Kafka error — %s", exc, exc_info=True)
        finally:
            if consumer:
                consumer.close()
                log.info("Kafka consumer closed.")

    def _poll_once(self, consumer: KafkaConsumer) -> None:
        log.info("Polling topic '%s' …", self._config.topic)
        batch_count = 0

        try:
            for message in consumer:
                batch_count += 1
                self._handle_message(message)
        except StopIteration:
            pass  # consumer_timeout_ms elapsed — normal exit from the iterator
        except KafkaError as exc:
            log.error("Error while consuming message — %s", exc, exc_info=True)

        log.info("Poll complete — %d message(s) processed.", batch_count)

    @staticmethod
    def _handle_message(message) -> None:
        email = message.key  # partition key set by Java publisher
        raw   = message.value

        log.debug(
            "Received message — topic=%s, partition=%d, offset=%d, key=%s",
            message.topic, message.partition, message.offset, email,
        )

        try:
            ordered_items = OrderedItems.from_json(raw)
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            log.error(
                "Failed to deserialise message at offset=%d — %s | raw=%s",
                message.offset, exc, raw[:200],
            )
            return

        # ── Print structured output ──────────────────────────────────────
        separator = "─" * 60
        print(f"\n{separator}")
        print(f"  Email   : {email}")
        print(f"  Topic   : {message.topic}  |  Partition: {message.partition}  |  Offset: {message.offset}")
        print(separator)
        print(ordered_items)
        print(separator)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    config = KafkaConfig()
    reader = OrderKafkaReader(config)

    # Graceful shutdown on SIGINT / SIGTERM
    def _shutdown(sig, _frame):
        print()
        log.info("Signal %s received — shutting down …", signal.Signals(sig).name)
        reader.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    reader.start()

    # Keep main thread alive
    log.info("Press Ctrl+C to stop.")
    signal.pause()


if __name__ == "__main__":
    main()