#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a trial of format <x> <y> <target|nontarget> to a veripair of format <0|1> <x> <y>")
    parser.add_argument("-i", "--input", type=str, help="The trial file to convert")
    parser.add_argument("-o", "--output", type=str, help="The output file to save")
    args = parser.parse_args()

    df = pd.read_csv(args.input, delimiter=" ", header=None, names=["target", "utt1", "utt2"])
    df["utt1"] = ['-'.join(i[:-4].split('/')) for i in df["utt1"]]
    df["utt2"] = ['-'.join(i[:-4].split('/')) for i in df["utt2"]]
    df.to_csv(args.output, sep=" ", header=None, columns=["target", "utt1",  "utt2"], index=False)