import argparse
import json
import os

import numpy as np
import pandas as pd


def softmax(x):
    shifted = x - np.max(x)
    exp_x = np.exp(shifted)
    return exp_x / exp_x.sum()


def build_schema(categorical_cols, numeric_cols):
    feature_cols = []
    if categorical_cols:
        feature_cols.extend(
            [
                {"name": name, "dtype": "str", "type": "categorical"}
                for name in categorical_cols
            ]
        )
    if numeric_cols:
        feature_cols.extend(
            [
                {"name": name, "dtype": "float", "type": "numeric"}
                for name in numeric_cols
            ]
        )
    return {
        "label_col": {"name": "label", "dtype": "int"},
        "feature_cols": feature_cols,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic wide CSV data for DCNv2 multiclass classification validation."
    )
    parser.add_argument(
        "--output_dir", required=True, help="Directory for generated files."
    )
    parser.add_argument(
        "--num_samples", type=int, default=5000, help="Number of synthetic samples."
    )
    parser.add_argument("--periods", type=int, default=6, help="Number of periods.")
    parser.add_argument(
        "--num_numeric_keys",
        type=int,
        default=24,
        help="Number of numeric base features.",
    )
    parser.add_argument(
        "--num_categorical_keys",
        type=int,
        default=12,
        help="Number of categorical base features.",
    )
    parser.add_argument(
        "--num_user_buckets",
        type=int,
        default=500,
        help="Number of synthetic user ids.",
    )
    parser.add_argument(
        "--num_classes", type=int, default=3, help="Number of label classes."
    )
    parser.add_argument("--seed", type=int, default=2026, help="Random seed.")
    args = parser.parse_args()

    if args.num_classes < 3:
        raise ValueError("num_classes must be at least 3 for multiclass generation.")

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    rows = []
    categorical_cols = ["user_id"]
    numeric_cols = []

    numeric_base = [
        "num_feat_{:02d}".format(idx) for idx in range(args.num_numeric_keys)
    ]
    categorical_base = [
        "cat_feat_{:02d}".format(idx) for idx in range(args.num_categorical_keys)
    ]

    for period_idx in range(args.periods):
        numeric_cols.append("rating_t{}".format(period_idx))
        for name in numeric_base:
            numeric_cols.append("{}_t{}".format(name, period_idx))
        for name in categorical_base:
            categorical_cols.append("{}_t{}".format(name, period_idx))

    numeric_weights = rng.normal(
        0.0, 0.18, size=(args.num_classes, args.periods, args.num_numeric_keys)
    )
    rating_weights = rng.normal(0.0, 0.22, size=(args.num_classes, args.periods))
    cat_weights = rng.normal(
        0.0, 0.08, size=(args.num_classes, args.periods, args.num_categorical_keys, 8)
    )
    class_bias = np.linspace(-0.35, 0.35, args.num_classes)

    for sample_idx in range(args.num_samples):
        row = {}
        user_bucket = int(rng.integers(args.num_user_buckets))
        row["user_id"] = "user_{:06d}".format(user_bucket)

        class_scores = class_bias.copy()
        class_scores += (user_bucket % max(args.num_classes, 3)) * np.linspace(
            -0.06, 0.06, args.num_classes
        )

        for period_idx in range(args.periods):
            rating_value = int(rng.integers(0, 10))
            row["rating_t{}".format(period_idx)] = float(rating_value)
            class_scores += rating_value * rating_weights[:, period_idx]

            for feature_idx, feature_name in enumerate(numeric_base):
                value = float(rng.normal(loc=0.2 * period_idx, scale=1.0))
                row["{}_t{}".format(feature_name, period_idx)] = value
                class_scores += value * numeric_weights[:, period_idx, feature_idx]

            for feature_idx, feature_name in enumerate(categorical_base):
                category_id = int(rng.integers(0, 8))
                row["{}_t{}".format(feature_name, period_idx)] = "c{}".format(
                    category_id
                )
                class_scores += cat_weights[:, period_idx, feature_idx, category_id]

        probabilities = softmax(class_scores)
        row["label"] = int(rng.choice(args.num_classes, p=probabilities))
        rows.append(row)

    output_df = pd.DataFrame(rows)

    wide_csv = os.path.join(args.output_dir, "wide.csv")
    schema_json = os.path.join(args.output_dir, "schema.json")
    output_df.to_csv(wide_csv, index=False)

    schema = build_schema(categorical_cols=categorical_cols, numeric_cols=numeric_cols)
    with open(schema_json, "w", encoding="utf-8") as output_file:
        json.dump(schema, output_file, ensure_ascii=True, indent=2)

    class_distribution = output_df["label"].value_counts(normalize=True).sort_index()

    print("Saved mock wide CSV to {}".format(wide_csv))
    print("Saved schema JSON to {}".format(schema_json))
    print("Output shape: {}".format(output_df.shape))
    print("Class distribution:")
    for class_id, ratio in class_distribution.items():
        print("  class {}: {:.4f}".format(int(class_id), float(ratio)))


if __name__ == "__main__":
    main()
