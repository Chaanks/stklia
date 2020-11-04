#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import argparse

def tar2int(s):
    assert s in ["target", "nontarget"], f"Unkown target type {s}"
    r = 1 if s == "target" else 0
    return r

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert a trial of format <x> <y> <target|nontarget> to a veripair of format <0|1> <x> <y>')
    parser.add_argument("-i", "--input", type=str, help="The trial file to convert")
    parser.add_argument("-o", "--output", type=str, help="The output file to save")
    args = parser.parse_args()

    df = pd.read_csv(args.input, delimiter=" ", header=None, names=["utt1", "utt2", "target"])
    df['target'] = df['target'].apply(tar2int)
    df.to_csv(args.output, sep=" ", header=None, columns=['target', 'utt1',  "utt2"], index=False)
