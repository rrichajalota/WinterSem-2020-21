#!/usr/bin/python3
"""
This script runs the entire pipeline and provides a user-interface to change the parameters.
"""
import argparse
from time import time

from cky_parser import *

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


def arg_parser():
    """ provides the user with options to train/test the HMM with different configurations. """
    parser = argparse.ArgumentParser()
    parser.add_argument("parser", type=bool,
                        help="If set to False, CKY recognizer runs, i.e. the algorithm will only tell if a sentence is grammatical or not."
                             "To compute the number of parse trees or draw trees, set parser=True."
                             "Note: When parser=True, by default, the algorithm will compute all the parse trees to retrieve the number of parse trees."
                             "Instead, to make the computation faster, also set --only_count=True.")
    parser.add_argument("--test", help='use this option to provide a file for testing the CKY parser.')
    parser.add_argument("--grammar", help="use this option to provide the CKY parser a grammar (file with '.cfg' extension)")
    parser.add_argument("--draw_trees", help="If specified, all possible parse trees are drawn. Note: this option can only be used if '--only_count' is False(or not specified).", action="store_true")
    parser.add_argument("--output_file_path", help="provide filename or filepath for saving the output file.")
    parser.add_argument("--only_count", type=bool,
                        help="If set to True, the algorithm simply counts the number of parse trees without actually computing all the parse trees. "
                             "Set this argument to True for faster computation."
                             "Note: if --draw_trees is specified, this argument MUST be set to False.")


    args = parser.parse_args()
    return args


def main():
    """
    runs the entire pipeline from loading the corpus to predicting the tagged output file.
    :return:
    """
    args = arg_parser()

    grammar = load_file(filename=args.grammar) if args.grammar else load_file("./atis/atis-grammar-cnf.cfg")
    test_sents = load_file(filename=args.test) if args.test else load_file("./atis/atis-test-sentences.txt")

    # TODO: check if the grammar is in chomsky_normal_form! if not, convert it to CNF and use that grammar instead. Also, save that grammar in results/

    output_file = args.output_file_path if args.output_file_path else 'results/result_compute_trees.txt'

    # check input
    check_input_args(args)

    with open(output_file, 'w') as f:
        start = time()
        for idx, sent in enumerate(test_sents):
            f.write(f" {' '.join(sent[0])}\t{cky_parser(sent[0], grammar, parser=args.parser, draw_tree=args.draw_trees, only_count=args.only_count)}\n")

        logging.info(f'Time taken by the parser when only_count is set to {args.only_count}: {time()-start}')


def check_input_args(args):
    if not args.parser and (args.draw_trees or args.only_count):
        raise ValueError(f'Set parser to False!')
    if args.draw_trees and args.only_count:
        raise ValueError(f'Both --draw_trees and --only_count cannot be set to True, simultaenously. See --help.')


if __name__ == '__main__':
    main()

