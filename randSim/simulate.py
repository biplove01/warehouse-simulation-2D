from dataclasses import dataclass, field
from typing import List
import json
import random


# ------------------------------------------------------------------
# Product Catalog
# item_code -> item_name
# ------------------------------------------------------------------

ITEM_CATALOG = {
    1: "T-Shirt",
    2: "Jeans",
    3: "Hoodie",
    4: "Jacket",
    5: "Sneakers",
    6: "Cap",
    7: "Polo Shirt",
    8: "Track Pants",
    9: "Blazer",
    10: "Sweatshirt",
    11: "Joggers",
    12: "Winter Coat",
    13: "Cargo Shorts",
    14: "Formal Shirt",
    15: "Scarf",
    16: "Gloves",
    17: "Sandals",
    18: "Beanie",
    19: "Socks",
    20: "Belt",
    21: "Watch",
    22: "Sunglasses",
    23: "Backpack",
    24: "Wallet",
}

SIZES = ["Small", "Medium", "Large", "XL", "XXL"]


# ------------------------------------------------------------------
# DTOs
# ------------------------------------------------------------------

@dataclass
class WarehouseItemData:
    """
    Mirrors com.ecomm.np.genevaecommerce.dto.WarehouseItemData.
    """
    order_tracer_code: int = 0
    item_name: str = ""
    item_code: int = 0
    size: str = ""
    quantity: int = 0

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
            f"WarehouseItemData("
            f"order_tracer_code={self.order_tracer_code}, "
            f"item_name='{self.item_name}', "
            f"item_code={self.item_code}, "
            f"size='{self.size}', "
            f"quantity={self.quantity})"
        )


@dataclass
class WarehouseData:
    """
    Mirrors com.ecomm.np.genevaecommerce.dto.WarehouseData.
    """
    o_id: int = 0
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
        item_lines = "\n".join(f"    {item}" for item in self.items)

        return (
            f"WarehouseData(\n"
            f"  o_id={self.o_id},\n"
            f"  items=[\n"
            f"{item_lines}\n"
            f"  ]\n"
            f")"
        )


# ------------------------------------------------------------------
# Random Warehouse Order Generator
# ------------------------------------------------------------------

def generate_random_warehouse_data(num_orders: int = 10) -> List[WarehouseData]:
    warehouse_orders = []

    for order_number in range(num_orders):
        order_id = 5001 + order_number

        items = []

        # each order has between 1 and 5 items
        item_count = random.randint(1, 5)

        for item_index in range(item_count):
            item_code = random.randint(1, 24)

            items.append(
                WarehouseItemData(
                    order_tracer_code=order_id * 100 + item_index + 1,
                    item_name=ITEM_CATALOG[item_code],
                    item_code=item_code,
                    size=random.choice(SIZES),
                    quantity=random.randint(1, 10),
                )
            )

        warehouse_orders.append(
            WarehouseData(
                o_id=order_id,
                items=items,
            )
        )

    return warehouse_orders


# ------------------------------------------------------------------
# Example Usage
# ------------------------------------------------------------------

if __name__ == "__main__":
    warehouse_data_list = generate_random_warehouse_data(10)

    for warehouse in warehouse_data_list:
        print(warehouse)
        print("-" * 80)