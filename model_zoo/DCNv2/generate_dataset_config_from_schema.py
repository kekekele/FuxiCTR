import argparse
import json


def format_feature_col(spec):
    return "{{name: %s, active: True, dtype: %s, type: %s}}" % (
        spec["name"],
        spec["dtype"],
        spec["type"],
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate a FuxiCTR DCNv2 dataset_config.yaml snippet from schema JSON."
    )
    parser.add_argument("--schema_json", required=True, help="Schema JSON path produced by the conversion script.")
    parser.add_argument("--dataset_id", required=True, help="Dataset id to use in dataset_config.")
    parser.add_argument("--data_root", required=True, help="data_root value for dataset_config.")
    parser.add_argument("--train_data", required=True, help="Train CSV path for dataset_config.")
    parser.add_argument("--valid_data", required=True, help="Valid CSV path for dataset_config.")
    parser.add_argument("--test_data", required=True, help="Test CSV path for dataset_config.")
    parser.add_argument("--output_yaml", default=None, help="Optional output yaml file. Prints to stdout if omitted.")
    args = parser.parse_args()

    with open(args.schema_json, "r", encoding="utf-8") as input_file:
        schema = json.load(input_file)

    feature_lines = ["        - " + format_feature_col(spec) for spec in schema["feature_cols"]]
    label_spec = schema["label_col"]

    yaml_text = "\n".join(
        [
            "{}:".format(args.dataset_id),
            "    data_root: {}".format(args.data_root),
            "    data_format: csv",
            "    train_data: {}".format(args.train_data),
            "    valid_data: {}".format(args.valid_data),
            "    test_data: {}".format(args.test_data),
            "    min_categr_count: 1",
            "    feature_cols:",
            *feature_lines,
            "    label_col: {name: %s, dtype: %s}" % (label_spec["name"], label_spec["dtype"]),
            "",
        ]
    )

    if args.output_yaml:
        with open(args.output_yaml, "w", encoding="utf-8") as output_file:
            output_file.write(yaml_text)
        print("Saved dataset config snippet to {}".format(args.output_yaml))
    else:
        print(yaml_text)


if __name__ == "__main__":
    main()