import argparse
import json
from collections import OrderedDict


def group_feature_cols(feature_cols):
    grouped_specs = OrderedDict()
    for spec in feature_cols:
        key = (spec["dtype"], spec["type"])
        grouped_specs.setdefault(key, []).append(spec["name"])
    return grouped_specs


def format_grouped_feature_spec(names, dtype, feature_type):
    joined_names = ",".join(['"{}"'.format(name) for name in names])
    return "{{name: [{}], active: True, dtype: {}, type: {}}}".format(
        joined_names, dtype, feature_type
    )


def format_feature_lines(grouped_items):
    if not grouped_items:
        return ["        []"]

    lines = []
    for idx, ((dtype, feature_type), names) in enumerate(grouped_items):
        spec_text = format_grouped_feature_spec(
            names=names, dtype=dtype, feature_type=feature_type
        )
        if idx == 0 and idx == len(grouped_items) - 1:
            lines.append("        [{}]".format(spec_text))
        elif idx == 0:
            lines.append("        [{},".format(spec_text))
        elif idx == len(grouped_items) - 1:
            lines.append("         {}]".format(spec_text))
        else:
            lines.append("         {},".format(spec_text))
    return lines


def main():
    parser = argparse.ArgumentParser(
        description="Generate a FuxiCTR DCNv2 dataset_config.yaml snippet from schema JSON."
    )
    parser.add_argument(
        "--schema_json",
        required=True,
        help="Schema JSON path produced by the conversion script.",
    )
    parser.add_argument(
        "--dataset_id", required=True, help="Dataset id to use in dataset_config."
    )
    parser.add_argument(
        "--data_root", required=True, help="data_root value for dataset_config."
    )
    parser.add_argument(
        "--train_data", required=True, help="Train CSV path for dataset_config."
    )
    parser.add_argument(
        "--valid_data", required=True, help="Valid CSV path for dataset_config."
    )
    parser.add_argument(
        "--test_data", required=True, help="Test CSV path for dataset_config."
    )
    parser.add_argument(
        "--output_yaml",
        default=None,
        help="Optional output yaml file. Prints to stdout if omitted.",
    )
    args = parser.parse_args()

    with open(args.schema_json, "r", encoding="utf-8") as input_file:
        schema = json.load(input_file)

    grouped_specs = group_feature_cols(schema["feature_cols"])
    grouped_items = list(grouped_specs.items())
    feature_lines = format_feature_lines(grouped_items)
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
            "    label_col: {name: %s, dtype: %s}"
            % (label_spec["name"], label_spec["dtype"]),
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
