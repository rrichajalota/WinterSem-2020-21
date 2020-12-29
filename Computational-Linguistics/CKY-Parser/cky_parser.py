import nltk
from collections import defaultdict
from nltk.tree import *
from time import time
import logging

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


def load_file(filename):
    data = nltk.data.load(filename)
    if not filename.endswith('.cfg'): # for test_file, grammars end with '.cfg'
        return nltk.parse.util.extract_test_sentences(data)
    return data


def build_grammar_dictionary(grammar): # preprocessing
    # iterate over all production rules and put rhs tuple as key and lhs as value for O(1) lookup.
    start = time()
    gram_dict = defaultdict(list)
    for prod in grammar.productions():
        gram_dict[prod.rhs()].append(prod.lhs())
    logging.info(f'time taken to build a dictionary out of the given grammar: {time() - start}')
    return gram_dict


def fetch_lhs(grammar_dict, rhs_tuple):
    return grammar_dict[rhs_tuple]


def parse_tree(chart, row, col, sym, only_count=True):
    start = time()
    backptr = chart[row][col][sym]

    # base condition - if string, we've reached a terminal.
    if type(backptr[0]) == type('str'):
        return 1 if only_count else [ImmutableTree(sym, backptr)]

    if only_count: num_trees = 0
    else: trees = []

    for tup in backptr:
        left, right = tup[0], tup[1] # right ~ bottom in cky chart
        lhs_trees = parse_tree(chart, left[0], left[1], left[2], only_count)
        rhs_trees = parse_tree(chart, right[0], right[1], right[2], only_count)

        if only_count:
            num_trees += lhs_trees * rhs_trees
        else:
            for t1 in lhs_trees: # combine all possible paths from left and right
                for t2 in rhs_trees:
                    trees.append(ImmutableTree(sym, [t1] + [t2]))

    return num_trees if only_count else trees


def cky_parser(words, grammar, grammar_dict, parser=False, draw_tree=False, only_count=True):
    n = len(words)

    chart = [[defaultdict(list) for j in range(n + 1)] for i in range(n)]

    for col in range(1, n + 1):  # left to right
        NT_list = fetch_lhs(grammar_dict, (words[col - 1],))
        for NT in NT_list:
            chart[col - 1][col][NT].append(words[col - 1])

        for row in range(col - 2, -1, -1):  # bottom-up
            for k in range(row + 1, col):   # k to mark the split of left and right phrasal windows
                for first_N in list(chart[row][k].keys()):
                    for second_N in list(chart[k][col].keys()):
                        NT_list = fetch_lhs(grammar_dict, (first_N, second_N))

                        for NT in NT_list: # backpointer information
                            chart[row][col][NT].append(((row, k, first_N), (k, col, second_N)))
    if parser is True:
        if only_count is True:
            if grammar.start() not in chart[0][n].keys(): return 0
            else: return parse_tree(chart, 0, n, grammar.start(), only_count=only_count)

        elif grammar.start() in chart[0][n].keys(): # otherwise, compute all parse trees
            tree = parse_tree(chart, 0, n, grammar.start(), only_count=only_count)

            if draw_tree is True:
                return tree

            return len(tree)

        return 0

    return grammar.start() in chart[0][n].keys()


if __name__ == '__main__':
    grammar = load_file("./atis/atis-grammar-cnf.cfg")
    test_sents = load_file("./atis/atis-test-sentences.txt")

    grammar_dict = build_grammar_dictionary(grammar)

    print(len(grammar.productions()))
    start = time()
    for idx, sent in enumerate(test_sents):
        cky_parser(sent[0], grammar, grammar_dict, True, False, False)
        if idx == 5:
            break
    print(time()-start)

    # gramma = nltk.data.load("./atis/atis-grammar-cnf.cfg")  # load the grammar
    # s = nltk.data.load("./atis/atis-test-sentences.txt")  # load raw sentences
    # t = nltk.parse.util.extract_test_sentences(s)
    # parser = nltk.parse.BottomUpChartParser(gramma)
    # # parse all test sentences
    #
    # with open('nltk_parser_results.txt', 'w') as f:
    #     for sent in t:
    #         try:
    #             f.write(f" {' '.join(sent[0])}\t{len(list(parser.chart_parse(sent[0])))}\n")
    #         except:
    #             f.write(f" {' '.join(sent[0])}\t0")
    #             continue