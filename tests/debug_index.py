import sys
import os

# ensure src in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.db import get_client, COLLECTION_NAME

client = get_client()

def list_all_vectors():
    print(f"Inspecting Collection: {COLLECTION_NAME}\n")

    # 1. get total count
    count_result = client.count(collection_name=COLLECTION_NAME)
    print(f"Total Points Stored: {count_result.count}")
    print("-" * 40)

    # 2. scroll points
    # fetch first 10
    points, next_page_offset = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=10,
        with_payload=True,
        with_vectors=False # set true to see 512 floats
    )

    if not points:
        print("Collection is empty!")
        return

    for point in points:
        print(f"ID: {point.id}")
        print(f"Payload (Specs): {point.payload}")
        print("-" * 20)

if __name__ == "__main__":
    list_all_vectors()