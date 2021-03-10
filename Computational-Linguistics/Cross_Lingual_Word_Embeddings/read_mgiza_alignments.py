#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @Author: Sangeet Sagar
# @Date:   2021-01-15 01:40:45
# @Email:  sasa00001@stud.uni-saarland.de
# @Organization: Universität des Saarlandes
# @Last Modified time: 2021-01-15 03:03:07


"""
Extract index-2-index alignments as obtained from the Mgiza alignments.
"""

import os
import sys
import re
import argparse
import numpy as np


def extract_alignments(alignment, myfile, args):
    
    alignments = alignment.split("})")[1:-1] # get token and its aligned indices
    for idx, item in enumerate(alignments):
        item = item.split(" ({") ## strip off parenthesis
        # e_word = item[0]
        if len(item) > 1:
            indices = item[1].split()
            if len(indices) != 0:
                for i in indices:
                    i = int(i)-1
                    myfile.write("%i-%i " % (i, idx))
    myfile.write("\n")
        
        
def main():
    """ main method """
    args = parse_arguments()
    # os.makedirs(args.out_dir, exist_ok=True)
    mgiza_f = args.mgiza_f
    outfile = str(args.out_dir) + "/mgiza.a"
    myfile = open(outfile, 'w')
    
    with open(mgiza_f, 'r') as f:
        for count, line in enumerate(f, start=1):
            #if count % 3 == 0:
            if line.startswith('NULL'):
                alignment = line.strip()
                extract_alignments(alignment, myfile, args)
    myfile.close()


def parse_arguments():
    """ parse arguments """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mgiza_f", help="path to alignment file generated from mgiza")
    parser.add_argument("out_dir", help="path to save generated alignments")
    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    main()

