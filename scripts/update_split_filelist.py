import os
import pandas as pd
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True, help='Path to the csv file')
    parser.add_argument('--train', type=int, required=True, help='Number of training samples')
    parser.add_argument('--val', type=int, required=True, help='Number of validation samples')
    parser.add_argument('--test', type=int, required=True, help='Number of testing samples')
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    assert 'Split' in df.columns, "FileName column not found in the csv file"

    if (args.train + args.val + args.test) > len(df):
        print(f"Warning: There are not enough samples in the csv file ({len(df)}) to split")
        print(f"into {args.train} training, {args.val} validation, and {args.test} testing samples ({args.train + args.val + args.test}).")
        print(f"Please adjust the number of samples or provide a csv file with more samples.")
        exit(1)
    elif (args.train + args.val + args.test) < len(df):
        print(f"Warning: There are more samples in the csv file ({len(df)}) than needed to split")
        print(f"into {args.train} training, {args.val} validation, and {args.test} testing samples ({args.train + args.val + args.test}).")
        print(f"Any extra samples will be be put into the \"EXTRA\" split.")

    df.loc[:args.train, 'Split'] = 'TRAIN'
    df.loc[args.train:args.train+args.val, 'Split'] = 'VAL'
    df.loc[args.train+args.val:args.train+args.val+args.test, 'Split'] = 'TEST'
    df.loc[args.train+args.val+args.test:, 'Split'] = 'EXTRA'

    df.to_csv(args.csv, index=False)
    print(f"Split the csv file into {args.train} training, {args.val} validation, and {args.test} testing samples.")
