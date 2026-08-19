import argparse
import json
from collections import defaultdict

import pandas as pd


def parse_csv_list(text, cast_func=str):
    if pd.isna(text):
        return []
    text = str(text).strip()
    if text == "":
        return []
    return [cast_func(item.strip()) for item in text.split(",") if item.strip() != ""]


def normalize_feature_type(feature_type):
    feature_type = str(feature_type).strip().lower()
    if feature_type in {
        "int",
        "integer",
        "float",
        "double",
        "long",
        "numeric",
        "number",
    }:
        return "numeric"
    if feature_type in {
        "str",
        "string",
        "category",
        "categorical",
        "id",
        "enum",
        "bool",
        "boolean",
    }:
        return "categorical"
    raise ValueError("Unsupported feature type: {}".format(feature_type))


def to_numeric_or_none(value):
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return None


def load_mapping(mapping_csv, key_col, type_col, name_col=None):
    mapping_df = pd.read_csv(mapping_csv)
    required_cols = {key_col, type_col}
    missing_cols = required_cols.difference(mapping_df.columns)
    if missing_cols:
        raise ValueError("Missing mapping columns: {}".format(sorted(missing_cols)))

    key_to_type = {}
    key_to_name = {}
    for _, row in mapping_df.iterrows():
        key = int(row[key_col])
        key_to_type[key] = normalize_feature_type(row[type_col])
        feature_name = (
            row[name_col]
            if name_col and name_col in row and pd.notna(row[name_col])
            else "key_{}".format(key)
        )
        key_to_name[key] = str(feature_name).strip() or "key_{}".format(key)
    return key_to_type, key_to_name


def build_feature_column_name(feature_name, period_idx):
    return "{}_t{}".format(feature_name, period_idx)


def transform_row(
    row,
    key_to_type,
    key_to_name,
    periods,
    user_col,
    key_col,
    time_col,
    value_col,
    rating_col,
    label_col,
    strict,
):
    user_id = row[user_col]
    keys = parse_csv_list(row[key_col], int)
    times = parse_csv_list(row[time_col], int)
    values = parse_csv_list(row[value_col], str)
    ratings = parse_csv_list(row[rating_col], int) if rating_col else []

    if len(keys) != len(times) or len(keys) != len(values):
        raise ValueError(
            "user_id={} has mismatched key/time/value lengths: {}/{}/{}".format(
                user_id, len(keys), len(times), len(values)
            )
        )

    if strict and len(ratings) not in (0, periods):
        raise ValueError(
            "user_id={} has {} ratings, expected 0 or {}".format(
                user_id, len(ratings), periods
            )
        )

    output = {
        user_col: str(user_id),
        label_col: int(row[label_col]),
    }

    if rating_col:
        for period_idx in range(periods):
            rating_value = ratings[period_idx] if period_idx < len(ratings) else None
            output["rating_t{}".format(period_idx)] = rating_value

    seen_pairs = set()
    unknown_keys = []
    invalid_periods = []
    for key, period_idx, raw_value in zip(keys, times, values):
        if key not in key_to_type:
            unknown_keys.append(key)
            continue
        if period_idx < 0 or period_idx >= periods:
            invalid_periods.append((key, period_idx))
            continue

        pair = (key, period_idx)
        if strict and pair in seen_pairs:
            raise ValueError(
                "user_id={} has duplicated (key={}, time={})".format(
                    user_id, key, period_idx
                )
            )
        seen_pairs.add(pair)

        feature_name = build_feature_column_name(key_to_name[key], period_idx)
        feature_type = key_to_type[key]
        if feature_type == "numeric":
            output[feature_name] = to_numeric_or_none(raw_value)
        else:
            output[feature_name] = str(raw_value).strip()

    return output, unknown_keys, invalid_periods


def build_schema(
    output_df, key_to_type, key_to_name, periods, user_col, label_col, has_rating
):
    feature_specs = []
    feature_specs.append({"name": user_col, "type": "categorical", "dtype": "str"})

    if has_rating:
        for period_idx in range(periods):
            feature_specs.append(
                {
                    "name": "rating_t{}".format(period_idx),
                    "type": "numeric",
                    "dtype": "float",
                }
            )

    for key in sorted(key_to_type.keys()):
        base_name = key_to_name[key]
        feature_type = key_to_type[key]
        dtype = "float" if feature_type == "numeric" else "str"
        for period_idx in range(periods):
            column_name = build_feature_column_name(base_name, period_idx)
            if column_name in output_df.columns:
                feature_specs.append(
                    {"name": column_name, "type": feature_type, "dtype": dtype}
                )

    return {
        "label_col": {"name": label_col, "dtype": "float"},
        "feature_cols": feature_specs,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert raw NPS CSV data to a DCNv2-friendly wide CSV."
    )
    parser.add_argument("--input_csv", required=True, help="Raw input CSV path.")
    parser.add_argument(
        "--mapping_csv",
        required=True,
        help="Mapping CSV path with key -> feature type.",
    )
    parser.add_argument("--output_csv", required=True, help="Wide output CSV path.")
    parser.add_argument(
        "--schema_json",
        default=None,
        help="Optional JSON path to dump inferred FuxiCTR feature schema.",
    )
    parser.add_argument(
        "--periods", type=int, default=6, help="Number of time periods."
    )
    parser.add_argument("--user_col", default="user_id", help="User id column name.")
    parser.add_argument("--key_col", default="key", help="Sparse key list column name.")
    parser.add_argument(
        "--time_col", default="time", help="Time index list column name."
    )
    parser.add_argument(
        "--value_col", default="value", help="Sparse value list column name."
    )
    parser.add_argument(
        "--rating_col",
        default="rating",
        help="Optional rating list column name. Missing column will be ignored.",
    )
    parser.add_argument("--label_col", default="label", help="Label column name.")
    parser.add_argument(
        "--mapping_key_col", default="key", help="Mapping CSV key column name."
    )
    parser.add_argument(
        "--mapping_type_col",
        default="feature_type",
        help="Mapping CSV feature type column name.",
    )
    parser.add_argument(
        "--mapping_name_col",
        default="feature_name",
        help="Optional mapping CSV feature name column name.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on duplicated key-time pairs or unexpected rating length.",
    )
    args = parser.parse_args()

    key_to_type, key_to_name = load_mapping(
        args.mapping_csv,
        key_col=args.mapping_key_col,
        type_col=args.mapping_type_col,
        name_col=args.mapping_name_col,
    )
    input_df = pd.read_csv(args.input_csv)

    required_cols = {
        args.user_col,
        args.key_col,
        args.time_col,
        args.value_col,
        args.label_col,
    }
    missing_cols = required_cols.difference(input_df.columns)
    if missing_cols:
        raise ValueError("Missing input columns: {}".format(sorted(missing_cols)))

    has_rating = bool(args.rating_col) and args.rating_col in input_df.columns

    output_rows = []
    total_unknown_keys = defaultdict(int)
    total_invalid_periods = defaultdict(int)
    for _, row in input_df.iterrows():
        transformed_row, unknown_keys, invalid_periods = transform_row(
            row=row,
            key_to_type=key_to_type,
            key_to_name=key_to_name,
            periods=args.periods,
            user_col=args.user_col,
            key_col=args.key_col,
            time_col=args.time_col,
            value_col=args.value_col,
            rating_col=args.rating_col if has_rating else None,
            label_col=args.label_col,
            strict=args.strict,
        )
        output_rows.append(transformed_row)
        for key in unknown_keys:
            total_unknown_keys[key] += 1
        for item in invalid_periods:
            total_invalid_periods[item] += 1

    output_df = pd.DataFrame(output_rows)

    front_columns = [args.user_col]
    if has_rating:
        front_columns.extend(
            ["rating_t{}".format(period_idx) for period_idx in range(args.periods)]
        )
    tail_columns = [args.label_col]
    middle_columns = sorted(
        [
            col
            for col in output_df.columns
            if col not in set(front_columns + tail_columns)
        ]
    )
    output_df = output_df[front_columns + middle_columns + tail_columns]

    output_df.to_csv(args.output_csv, index=False)

    if args.schema_json:
        schema = build_schema(
            output_df=output_df,
            key_to_type=key_to_type,
            key_to_name=key_to_name,
            periods=args.periods,
            user_col=args.user_col,
            label_col=args.label_col,
            has_rating=has_rating,
        )
        with open(args.schema_json, "w", encoding="utf-8") as output_file:
            json.dump(schema, output_file, ensure_ascii=True, indent=2)

    print("Saved wide CSV to {}".format(args.output_csv))
    print("Output shape: {}".format(output_df.shape))
    if args.schema_json:
        print("Saved schema JSON to {}".format(args.schema_json))
    if total_unknown_keys:
        print(
            "Warning: ignored unknown keys: {}".format(
                dict(sorted(total_unknown_keys.items()))
            )
        )
    if total_invalid_periods:
        print(
            "Warning: ignored invalid (key, time) pairs: {}".format(
                dict(sorted(total_invalid_periods.items()))
            )
        )


if __name__ == "__main__":
    main()
