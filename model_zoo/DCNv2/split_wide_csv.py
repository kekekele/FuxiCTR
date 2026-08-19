import argparse

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Split a transformed wide CSV into train/test, with valid equal to test."
    )
    parser.add_argument("--input_csv", required=True, help="Wide input CSV path.")
    parser.add_argument("--train_csv", required=True, help="Output train CSV path.")
    parser.add_argument("--test_csv", required=True, help="Output test CSV path.")
    parser.add_argument(
        "--test_ratio", type=float, default=0.3, help="Test split ratio. Default: 0.3"
    )
    parser.add_argument(
        "--seed", type=int, default=2026, help="Random seed for reproducible splitting."
    )
    parser.add_argument(
        "--label_col",
        default="label",
        help="Optional label column used for stratified sampling when available.",
    )
    args = parser.parse_args()

    if not 0 < args.test_ratio < 1:
        raise ValueError("test_ratio must be in (0, 1).")

    df = pd.read_csv(args.input_csv)
    test_df = None
    if args.label_col in df.columns and df[args.label_col].nunique(dropna=False) > 1:
        test_parts = []
        train_parts = []
        for _, group_df in df.groupby(args.label_col, dropna=False):
            group_test = group_df.sample(frac=args.test_ratio, random_state=args.seed)
            group_train = group_df.drop(group_test.index)
            test_parts.append(group_test)
            train_parts.append(group_train)
        train_df = pd.concat(train_parts).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        test_df = pd.concat(test_parts).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    else:
        test_df = df.sample(frac=args.test_ratio, random_state=args.seed)
        train_df = df.drop(test_df.index).reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

    train_df.to_csv(args.train_csv, index=False)
    test_df.to_csv(args.test_csv, index=False)

    print("Saved train CSV to {}".format(args.train_csv))
    print("Saved test CSV to {}".format(args.test_csv))
    print("Use the same test CSV path as valid_data in dataset_config.")
    print("Train shape: {}".format(train_df.shape))
    print("Test shape: {}".format(test_df.shape))


if __name__ == "__main__":
    main()