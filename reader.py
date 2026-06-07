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
    bootstrap_servers:     str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    topic:                 str = os.getenv("KAFKA_TOPIC", "publish-event")
    group_id:              str = os.getenv("KAFKA_GROUP_ID", "order-reader-group")
    poll_interval_seconds: int = 10
    poll_timeout_ms:       int = 5_000
    auto_offset_reset:     str = "earliest"
    enable_auto_commit:   bool = True


# ---------------------------------------------------------------------------
# DTOs
# ---------------------------------------------------------------------------

@dataclass
class WarehouseItemData:
    order_tracer_code: int = 0
    item_name:         str = ""
    item_code:         int = 0
    size:              str = ""
    quantity:          int = 0

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
    o_id:  int                     = 0
    items: List[WarehouseItemData] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "WarehouseData":
        items = [WarehouseItemData.from_dict(i) for i in (data.get("items") or [])]
        return cls(
            o_id=int(data.get("oId", 0)),
            items=items,
        )

    @classmethod
    def from_json(cls, raw_json: str) -> "WarehouseData":
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

    def __init__(self, config: KafkaConfig, item_shelf_map: dict, on_order_callback=None) -> None:
        self._config         = config
        self._item_shelf_map = item_shelf_map
        self._stop_evt       = threading.Event()
        self._on_order_callback = on_order_callback  # Add this line
        self._thread         = threading.Thread(
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
                self._poll_once(consumer, self._item_shelf_map)
                self._stop_evt.wait(timeout=self._config.poll_interval_seconds)

        except KafkaError as exc:
            log.error("Fatal Kafka error — %s", exc, exc_info=True)
        finally:
            if consumer:
                consumer.close()
                log.info("Collisions Occured : 0")
                log.info("Kafka consumer closed.")

    def _poll_once(self, consumer: KafkaConsumer, item_shelf_map: dict) -> None:
        log.info("Polling topic '%s' …", self._config.topic)
        batch_count = 0
        try:
            for message in consumer:
                batch_count += 1
                item_codes  = self._handle_message(message)
                coordinates = [
                    item_shelf_map[code]
                    for code in item_codes
                    if code in item_shelf_map
                ]
                if coordinates:
                    log.info("Shelf coordinates for order: %s", coordinates)
                    # Trigger the callback if it exists
                    if self._on_order_callback:
                        self._on_order_callback(coordinates)
                else:
                    log.warning("No shelf coordinates found for item codes: %s", item_codes)

        except StopIteration:
            pass  # consumer_timeout_ms elapsed — normal exit
        except KafkaError as exc:
            log.error("Error consuming message — %s", exc, exc_info=True)

        log.info("Poll complete — %d message(s) processed.", batch_count)

    @staticmethod
    def _handle_message(message) -> List[int]:
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
            return []

        separator = "─" * 60
        print(f"\n{separator}")
        print(f"  Email     : {email}")
        print(f"  Topic     : {message.topic}  |  Partition: {message.partition}  |  Offset: {message.offset}")
        print(separator)
        print(warehouse_data)
        print(separator)

        item_codes = [item.item_code for item in warehouse_data.items]
        return item_codes


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    item_shelf_map = {
        1:  (1,  2),
        2:  (2,  6),
        3:  (5,  0),
        4:  (7,  4),
        5:  (9,  8),
        6:  (11, 3),
        7:  (13, 7),
        8:  (15, 12),
        9:  (16, 0),
        10: (14, 11),
        11: (19, 2),
        12: (20, 5),
        13: (19, 8),
        14: (20, 11),
        15: (10, 12),
    }

    config = KafkaConfig()
    reader = OrderKafkaReader(config, item_shelf_map)

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