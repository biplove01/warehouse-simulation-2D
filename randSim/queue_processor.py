from queue import Queue

# Import from your first file
from simulate import generate_random_warehouse_data


SIZE_CODE = {
    "Small": 1,
    "Medium": 2,
    "Large": 3,
    "XL": 4,
    "XXL": 5,
}


def process_orders(order_queue: Queue):
    while not order_queue.empty():
        warehouse_data = order_queue.get()

        print(f"\nProcessing Order: {warehouse_data.o_id}")

      
        warehouse_item_list = warehouse_data.items

        for item in warehouse_item_list:
            item_code = item.item_code
            item_name = item.item_name
            size = item.size
            quantity = item.quantity

            size_code = SIZE_CODE[size]
            index = item_code * 5 + size_code

            for _ in range(quantity):
                print(
                    f"ItemName: {item_name:<15} "
                    f"ItemCode: {item_code:<2} "
                    f"Size: {size:<6} "
                    f"Index: {index}"
                )

        order_queue.task_done()


def main():
    warehouse_data_list = generate_random_warehouse_data(10)

    print("Orders generated:", len(warehouse_data_list))

    order_queue = Queue()

    for warehouse_data in warehouse_data_list:
        order_queue.put(warehouse_data)

    print("Queue size:", order_queue.qsize())

    process_orders(order_queue)


if __name__ == "__main__":
    main()