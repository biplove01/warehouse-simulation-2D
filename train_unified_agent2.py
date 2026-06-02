"""
reader.py — Kafka consumer for the 'publish-event' topic.

Reads messages every 10 seconds (via a scheduled polling thread), deserialises
the JSON payload into typed Python dataclasses that mirror the Java DTOs
published by OrderPublisherImpl, and prints a structured summary to stdout.

Payload schema (from Java DTOs):
    WarehouseData:
        oId          : int
        items        : List[WarehouseItemData]

    WarehouseItemData:
        orderTracerCode : int
        itemName        : str
        itemCode        : int
        size            : str
        quantity        : int

Dependencies:
    pip install kafka-python-ng

Environment variables (or edit KafkaConfig below):
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
    bootstrap_servers:    str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    topic:                str = os.getenv("KAFKA_TOPIC", "order-publish")
    group_id:             str = os.getenv("KAFKA_GROUP_ID", "order-reader-group")
    poll_interval_seconds: int = 10
    poll_timeout_ms:       int = 5_000
    auto_offset_reset:    str = "earliest"
    enable_auto_commit:  bool = True


# ---------------------------------------------------------------------------
# DTOs — mirror Java WarehouseData / WarehouseItemData exactly
# ---------------------------------------------------------------------------

@dataclass
class WarehouseItemData:
    """
    Mirrors com.ecomm.np.genevaecommerce.dto.WarehouseItemData.
    All fields are primitives/strings — no nested entities, no date types.
    """
    order_tracer_code: int  = 0
    item_name:         str  = ""
    item_code:         int  = 0
    size:              str  = ""
    quantity:          int  = 0

    @classmethod
    def from_dict(cls, data: dict) -> "WarehouseItemData":
        return cls(
            order_tracer_code=int(data.get("orderTracerCode", 0)),
            item_name=str(data.get("itemName", "")),
            item_code=int(data.get("itemCode", 0)),
            size=str(data.get("size", "")),
            quantity=int(data.get("quantity", 0)),
        )

    def __str__(self) -> str:
        return (
            f"    WarehouseItemData(\n"
            f"      orderTracerCode = {self.order_tracer_code}\n"
            f"      itemName        = {self.item_name!r}\n"
            f"      itemCode        = {self.item_code}\n"
            f"      size            = {self.size!r}\n"
            f"      quantity        = {self.quantity}\n"
            f"    )"
        )


@dataclass
class WarehouseData:
    """
    Mirrors com.ecomm.np.genevaecommerce.dto.WarehouseData.
    This is the exact payload published to Kafka by OrderPublisherImpl
    via WarehouseData.buildFromOrder(orderedItems).
    """
    o_id:  int                      = 0
    items: List[WarehouseItemData]  = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "WarehouseData":
        """
        Deserialises a raw dict into a WarehouseData instance, recursively
        mapping each item in the 'items' list to a WarehouseItemData.
        Equivalent to: objectMapper.readValue(json, WarehouseData.class)
        """
        items = [WarehouseItemData.from_dict(i) for i in (data.get("items") or [])]
        return cls(
            o_id=int(data.get("oId", 0)),
            items=items,
        )

    @classmethod
    def from_json(cls, raw_json: str) -> "WarehouseData":
        """Entry point — accepts the raw JSON string straight from Kafka."""
        return cls.from_dict(json.loads(raw_json))

    def __str__(self) -> str:
        item_lines = (
            "\n".join(str(i) for i in self.items)
            if self.items
            else "    (no items)"
        )
        return (
            f"WarehouseData(\n"
            f"  oId   = {self.o_id}\n"
            f"  items =\n{item_lines}\n"
            f")"
        )


# ---------------------------------------------------------------------------
# Kafka consumer
# ---------------------------------------------------------------------------

class OrderKafkaReader:
    """
    Polls the Kafka topic on a background daemon thread every
    `config.poll_interval_seconds` seconds, deserialises each message
    into a WarehouseData instance, and prints it to stdout.
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
            pass  # consumer_timeout_ms elapsed — normal, expected exit
        except KafkaError as exc:
            log.error("Error consuming message — %s", exc, exc_info=True)

        log.info("Poll complete — %d message(s) processed.", batch_count)

    @staticmethod
    def _handle_message(message) -> None:
        email = message.key
        raw   = message.value

        log.debug(
            "Received — topic=%s, partition=%d, offset=%d, key=%s",
            message.topic, message.partition, message.offset, email,
        )

        try:
            warehouse_data = WarehouseData.from_json(raw)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            log.error(
                "Failed to deserialise message at offset=%d — %s | raw=%s",
                message.offset, exc, raw[:300],
            )
            return

        separator = "─" * 60
        print(f"\n{separator}")
        print(f"  Email     : {email}")
        print(f"  Topic     : {message.topic}  |  Partition: {message.partition}  |  Offset: {message.offset}")
        print(separator)
        print(warehouse_data)
        print(separator)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    config = KafkaConfig()
    reader = OrderKafkaReader(config)

    def _shutdown(sig, _frame):
        print()
        log.info("Signal %s received — shutting down …", signal.Signals(sig).name)
        reader.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    reader.start()
    log.info("Press Ctrl+C to stop.")
    signal.pause()


if __name__ == "__main__":
    main()