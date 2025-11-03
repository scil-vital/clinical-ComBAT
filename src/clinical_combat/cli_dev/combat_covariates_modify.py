#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse

import pandas as pd

from clinical_combat.utils.scilpy_utils import add_overwrite_arg


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("input_csv", help="Path to csv")
    p.add_argument("--column_name", help="Column name to replace")
    p.add_argument("--new_value", help="New value")
    p.add_argument("--new_column_name", help="New column name")
    p.add_argument("--output", help="Output")

    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    df[args.new_column_name] = df[args.column_name].copy()
    df[args.column_name] = args.new_value

    df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
