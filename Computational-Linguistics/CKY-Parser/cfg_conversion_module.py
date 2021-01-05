import nltk
from collections import Counter, defaultdict
from nltk.grammar import *
import logging
import pprint

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


def get_key_from_value(my_dict, to_find):
    for k,list_v in my_dict.items():
        for val in list_v:
            if val==to_find: return k
    return None


def get_interim_rhs(gram_dict, lhs):  # lhs tuple (NT,)
    rhs_list = gram_dict[lhs[0]]
    # base condition
    if all((len(rhs) == 1 and type(rhs[0]) == type('s')) or len(rhs) > 1 for rhs in rhs_list):
        return rhs_list

    valid_rhs = []
    for rhs in rhs_list:
        if len(rhs) == 1 and type(rhs[0]) != type('s'):
            valid_rhs.extend(get_interim_rhs(gram_dict, rhs))
        else:
            valid_rhs.append(rhs)
    return valid_rhs


def build_grammar_dict(nltk_grammar):
    # put all the rules of orig. orig_grammar in a dictionary
    orig_gram_dict = defaultdict(list)  # NT key -> list of RHS tuples
    for prod in nltk_grammar.productions():
        orig_gram_dict[prod.lhs()].append(prod.rhs())
    return orig_gram_dict


def convert_to_CNF(given_grammar):
    '''
    converts any orig_grammar to chomsky normal form.
    '''
    orig_gram_dict = build_grammar_dict(given_grammar)
    interim_gram = remove_single_RHS_NTs(orig_gram_dict, given_grammar)
    # pprint.pprint((interim_gram))
    return binarize_rules(interim_gram)


def binarize_rules(interim_gram):
    cnf_gram = defaultdict(list)

    for lhs, list_of_rhs in list(interim_gram.items()):
        for rhs in list_of_rhs:
            if len(rhs) > 2:
                while len(rhs) != 2:
                    l1, l2 = rhs[0], rhs[1] # take the 2 leftmost nodes and combine them
                    dum_key = str(l1) + '_' + str(l2)
                    # TODO: look for (l1,l2) in the dict_values and see if any corresponding key exists.
                    #  if it does, add it to rhs[2:] in line 57
                    if len(cnf_gram[Nonterminal(dum_key)]) == 0:
                        cnf_gram[Nonterminal(dum_key)].append((l1, l2))
                    rhs = (Nonterminal(dum_key),) + rhs[2:]

            cnf_gram[lhs].append(rhs) # add the lhs and the remaining two NTs in the cnf orig_grammar

    return (cnf_gram)


def remove_single_RHS_NTs(orig_gram_dict, orig_grammar):
    # by the end of this loop S->[NT1, NT2, NT3,..] or S->T, NT-> T, NT-> NT1, NT2,NT3...
    interim_gram = defaultdict(list)

    for lhs, list_of_rhs in list(orig_gram_dict.items()):

        if lhs == orig_grammar.start() or not all(
                (len(rhs) == 1 and type(rhs) == type('s')) or len(rhs) > 1 for rhs in list_of_rhs):
            for rhs in list_of_rhs:
                if len(rhs) == 1 and type(rhs[0]) != type('str'):  # i.e. a non-terminal
                    interim_rhs = get_interim_rhs(orig_gram_dict, rhs)
                    interim_gram[lhs].extend(interim_rhs)

                else:
                    interim_gram[lhs].append(rhs)

            if len(interim_gram[lhs]) == 0:
                print(lhs, list_of_rhs)
        else:
            interim_gram[lhs] = list_of_rhs

    return interim_gram


def write_grammar_to_file(cnf_gram, output_filepath):
    new_rule = ""
    for lhs, rhs_list in cnf_gram.items():
        for rhs in rhs_list:
            if len(rhs) == 1: new_rule += str(lhs) + ' -> "' + rhs[0] + '"\n'
            else: new_rule += str(lhs) + ' -> ' + ' '.join([str(r) for r in rhs]) + '\n'

    with open(output_filepath, 'w') as f:
        for rule in new_rule.split('\n'):
            f.write(rule + '\n')


def is_grammar_cnf(cnf_gram):
    ## check if orig_grammar is in CNF
    for lhs, list_of_rhs in cnf_gram.items():
        for rhs in list_of_rhs:
            if len(rhs) == 2 or (len(rhs) == 1 and type(rhs[0]) == type('str')):
                continue
            else:
                logging.error(f'The orig_grammar is not in Chomsky Normal Form due to the rule {lhs} -> {rhs}')
                return False
    return True


def cfg_to_cnf(cfg_grammar_path="grammars/large_grammars/atis.cfg", outfile='results/generated_cnf_grammar.cfg'):
    '''
    controller fn. that converts grammar into CNF and also checks if the resultant grammar is in CNF and saves it to the given path
    '''
    nltk_grammar = nltk.data.load(cfg_grammar_path)  # load the original grammar
    cnf_gram_dict = convert_to_CNF(given_grammar=nltk_grammar)
    if is_grammar_cnf(cnf_gram_dict):
        write_grammar_to_file(cnf_gram_dict, output_filepath=outfile)
    

if __name__ == '__main__':
    from cky_parser import *
    cfg_to_cnf(cfg_grammar_path="./atis/atis-grammar-original.cfg", outfile='results/generated_cnf_grammar.cfg')

    grammar = load_file('results/generated_cnf_grammar.cfg')
    logging.info(f'size of CFG grammar: {len(grammar.productions())}')
    
    test_sents = load_file("./atis/atis-test-sentences.txt")
    grammar_dict = build_grammar_dictionary(grammar)

    logging.info(f'size of the CNF orig_grammar dictionary: {len(grammar_dict)}')

    result = cky_parser(['prices'], grammar, grammar_dict, parser=True, draw_tree=False,
                        only_count=True)
    print(['price'], result)

    # for idx, sent in enumerate(test_sents):
    #     if idx == 10: break
    #     result = cky_parser(sent[0], grammar, grammar_dict, parser=True, draw_tree=False,
    #                         only_count=True)
    #     print(sent[0], result)



