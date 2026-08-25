import argparse
import json

import pandas as pd


def normalize_value_size(value):
    if value == -1:
        return "numeric"
    return "categorical"


def main():
    parser = argparse.ArgumentParser(
        description="Generate mapping CSV for DCNv2 conversion from a schema JSON file."
    )
    parser.add_argument("--schema_json", required=True, help="Input schema JSON path.")
    parser.add_argument("--output_csv", required=True, help="Output mapping CSV path.")
    args = parser.parse_args()

    with open(args.schema_json, "r", encoding="utf-8") as input_file:
        schema = json.load(input_file)

    value_size_list = schema.get("value_size_list")
    key_size = int(schema.get("key_size"))
    key_parts = schema.get("key_parts")

    if not isinstance(value_size_list, list) or len(value_size_list) == 0:
        raise ValueError("value_size_list must be a non-empty list.")
    if not isinstance(key_parts, list) or len(key_parts) == 0:
        raise ValueError("key_parts must be a non-empty list.")
    if key_size <= 0:
        raise ValueError("key_size must be a positive integer.")

    rows = []
    value_type_count = len(value_size_list)
    feature_name_count = len(key_parts)
    for key in range(key_size):
        feature_type = normalize_value_size(value_size_list[key % value_type_count])
        feature_name = key_parts[key % feature_name_count]
        rows.append(
            {
                "key": key,
                "feature_name": feature_name,
                "feature_type": feature_type,
            }
        )

    mapping_df = pd.DataFrame(rows)
    mapping_df.to_csv(args.output_csv, index=False)

    print("Saved mapping CSV to {}".format(args.output_csv))
    print("Number of keys: {}".format(len(mapping_df)))


if __name__ == "__main__":
    main()
