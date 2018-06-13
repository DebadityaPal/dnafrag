#!/usr/bin/env python

from __future__ import absolute_import

import argparse
import os

from .core import write_fragbed


def main():
    parser = argparse.ArgumentParser("dnafrag-from-fragbed")
    parser.add_argument("fragbed", type=os.path.abspath)
    parser.add_argument("outdir", type=os.path.abspath)
    parser.add_argument("genome", type=os.path.abspath)
    parser.add_argument("--max-fraglen", type=int, default=400)
    args = parser.parse_args()

    write_fragbed(args.fragbed, args.outdir, args.genome, args.max_fraglen)


if __name__ == "__main__":
    main()
