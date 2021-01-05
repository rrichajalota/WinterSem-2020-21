#!/usr/bin/python3
"""
This script runs the entire pipeline and provides a user-interface to change the parameters.
"""
import argparse
import sys
from nltk.draw.tree import draw_trees
from cfg_conversion_module import *

from cky_parser import *

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


def arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--parser", dest='parser', default=False, action='store_true',
                        help="If the argument is not specified, by default, CKY recognizer runs, i.e. the algorithm will only tell if a sentence is grammatical or not."
                             "To compute the number of parse trees or draw trees, call './main.py --parser' or './main.py --parser --draw_trees', respectively."
                             "Note: When --parser is set, by default, the algorithm will compute all the parse trees to retrieve the number of parse trees."
                             "Instead, to make the computation faster, also set --only_count.")
    parser.add_argument("--test", help='use this option to provide a file for testing the CKY parser.')
    parser.add_argument("--grammar", help="use this option to provide the CKY parser a grammar (file with '.cfg' extension)")
    parser.add_argument("--draw_trees", dest='draw_trees', default=False,
                        help="If specified, all possible parse trees are drawn for the first 10 input sentences. "
                             "Note: this option can only be used if '--only_count' is False (i.e. not specified).", action="store_true")
    parser.add_argument("--output_file_path", help="provide filename or filepath for saving the output file.")
    parser.add_argument("--only_count", default=False, dest='only_count', action='store_true',
                        help="If specified, the algorithm simply counts the number of parse trees without actually computing all the parse trees. "
                             "Set this argument for faster computation with --parser."
                             "Note: if --draw_trees is specified, this argument MUST be set to False i.e. must not be specified.")


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
    output_file = args.output_file_path if args.output_file_path else 'results/CKY_result.txt'

    # TODO: check if the orig_grammar is in chomsky_normal_form! if not, convert it to CNF and use that orig_grammar instead. Also, save that orig_grammar in results/

    # check input
    check_input_args(args)

    # optimiztaion step: build a dictionary out of CNF-orig_grammar
    grammar_dict = build_grammar_dictionary(grammar)
    logging.info(f'size of the CNF grammar look-up dictionary: {len(grammar_dict)}')
    logging.info(f'args.parser: {args.parser}')

    if args.draw_trees is True:
        sys.stdout = open(output_file, 'w')
        for idx, sent in enumerate(test_sents):
            if idx == 10: break
            result = cky_parser(sent[0], grammar, grammar_dict, parser=args.parser, draw_tree=args.draw_trees,
                                only_count=args.only_count)
            try:
                print(f"({idx})\t{' '.join(sent[0])}\t{len(result)}\n")
                for t in result:
                    Tree.fromstring(str(t)).pretty_print()
                print("x--------------------------------------------------------------------------------------------x\n\n")
            except TypeError:
                print(f"({idx})\t{' '.join(sent[0])}\t 0 parse trees.")
                continue

        sys.stdout.close()

    else:
        with open(output_file, 'w') as f:
            start = time()
            for idx, sent in enumerate(test_sents):
                f.write(f" {' '.join(sent[0])}\t{cky_parser(sent[0], grammar, grammar_dict, parser=args.parser, draw_tree=args.draw_trees, only_count=args.only_count)}\n")

            f.write(f"\n\n---------------------\n\nTime taken by the parser when only_count is set to {args.only_count}: {time()-start}")


def check_input_args(args):
    if args.parser is False and (args.draw_trees or args.only_count):
        raise ValueError(f'Set parser to False!')
    if args.draw_trees is True and args.only_count is True:
        raise ValueError(f'Both --draw_trees and --only_count cannot be set to True, simultaenously. See --help.')


if __name__ == '__main__':
    main()

