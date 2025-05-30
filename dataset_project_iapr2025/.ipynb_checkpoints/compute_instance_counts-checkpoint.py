#!/usr/bin/env python
import json
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Count number of instances per category in a COCO-format annotation file."
    )
    parser.add_argument(
        "coco_json",
        help="Path to the COCO JSON file (e.g. train_coco_dataset.json)"
    )
    args = parser.parse_args()

    # Load the annotations
    with open(args.coco_json, 'r') as f:
        coco = json.load(f)

    # Build a mapping category_id -> category_name
    cat_id_to_name = {cat['id']: cat['name'] for cat in coco['categories']}

    # Initialize counts
    counts = {cid: 0 for cid in cat_id_to_name}

    # Tally up
    for ann in coco['annotations']:
        cid = ann['category_id']
        if cid in counts:
            counts[cid] += 1

    # Print results
    print("Instance counts per category:")
    for cid in sorted(counts):
        print(f"  {cat_id_to_name[cid]:<20s} {counts[cid]}")

if __name__ == "__main__":
    main()
