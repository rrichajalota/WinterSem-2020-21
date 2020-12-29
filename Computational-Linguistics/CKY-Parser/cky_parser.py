import nltk
from collections import defaultdict
from nltk.tree import *

import logging

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


def load_file(filename):
    data = nltk.data.load(filename)
    if not filename.endswith('.cfg'): # for test_file, grammars end with '.cfg'
        return nltk.parse.util.extract_test_sentences(data)
    return data


def fetch_lhs(grammar, rhs_tuple):
    all_prods = []
    filtered_productions = grammar.productions(rhs=rhs_tuple[0])

    if len(rhs_tuple) == 1:  # it's a terminal
        for prod in filtered_productions:
            all_prods.append(prod.lhs())
        return all_prods

    # print(filtered_productions)
    for a_production in filtered_productions:
        if rhs_tuple[1] == a_production.rhs()[1]:
            all_prods.append(a_production.lhs())
    return all_prods


def parse_tree(chart, row, col, sym, only_count=True):
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
        # for t in trees:
        #    print(f' after append: {Tree.fromstring(str(t)).pretty_print()}')
    return num_trees if only_count else trees


def cky_parser(words, grammar, parser=False, draw_tree=False, only_count=True):
    n = len(words)
    #print(n)
    chart = [[defaultdict(list) for j in range(n + 1)] for i in range(n)]

    for col in range(1, n + 1):  # left to right
        NT_list = fetch_lhs(grammar, (words[col - 1],))
        for NT in NT_list:
            chart[col - 1][col][NT].append(words[col - 1])
            # print(NT, chart[col-1][col][NT][0])

        for row in range(col - 2, -1, -1):  # bottom-up
            for k in range(row + 1, col):   # k to mark the split of left and right phrasal windows
                for first_N in list(chart[row][k].keys()):
                    for second_N in list(chart[k][col].keys()):
                        #print(f'chart[{row}][{k}] : {chart[row][k]}')
                        #print(f'chart[{k}][{col}] : {chart[k][col]}')
                        NT_list = fetch_lhs(grammar, (first_N, second_N))
                        # print(NT_list)
                        for NT in NT_list: # backpointer information
                            chart[row][col][NT].append(((row, k, first_N), (k, col, second_N)))
                            # print(NT, chart[row][col][NT])
    if parser:
        if only_count:
            if grammar.start() not in chart[0][n].keys(): return 0
            else: return parse_tree(chart, 0, n, grammar.start(), only_count=True)

        elif grammar.start() in chart[0][n].keys(): # otherwise, compute all parse trees
            tree = parse_tree(chart, 0, n, grammar.start(), only_count=False)
            if draw_tree:
                for t in tree:
                    Tree.fromstring(str(t)).pretty_print()
            return len(tree)

        return 0

    return grammar.start() in chart[0][n].keys()


if __name__ == '__main__':
    grammar = load_file("./atis/atis-grammar-cnf.cfg")
    test_sents = load_file("./atis/atis-test-sentences.txt")

    print(len(grammar.productions()))

    with open('result_compute_trees.txt', 'w') as f:
        for idx, sent in enumerate(test_sents):
            f.write(f" {' '.join(sent[0])}\t{cky_parser(sent[0], grammar, True, False, True)}\n")

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