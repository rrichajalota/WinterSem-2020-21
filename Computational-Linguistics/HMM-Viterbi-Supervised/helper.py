
from nltk.corpus.reader import ConllCorpusReader
import logging

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

def read_corpus(path='de-utb/de-train.tt', columntypes=('words', 'pos')):  # by default reads the training data
    return ConllCorpusReader('.', path, columntypes=columntypes)


def obj_dict_index(list_obj):
    return dict(zip(list_obj, range(0,len(list_obj))))


def record_states_from_corpus(reader_corpus_obj):
    '''
    :param reader_corpus_obj: of type nltk.corpus.reader.conll.ConllCorpusReader
    :return: unique list of POS-tags from the corpus, state2index dictionary e.g. {'NN': 0, 'PN': 4,...}
    '''
    states = list(set([tup[1] for tup in reader_corpus_obj.tagged_words()])) # list of unique POS tags
    return states, obj_dict_index(states)


def record_obs_from_corpus(reader_corpus_obj):
    '''
    :param reader_corpus_obj: of type nltk.corpus.reader.conll.ConllCorpusReader
    :return: list of unique words found in the training corpus, word2index dict e.g. {'der': 12, 'schon': 0,...}
    '''
    obs = list(set([tup[0] for tup in reader_corpus_obj.tagged_words()])) # list of unique observations
    return obs, obj_dict_index(obs)


def write_tags_to_file(predicted_tags:list, test_file='./de-utb/de-test.t', tagged_file='de-utb/de-tagged.tt'):
    with open(test_file) as inp_file, open(tagged_file, 'w') as op_file:
        i, j = 0, 0
        for line in inp_file:
            if line == '\n': # delimiter marks the end of sentence
                i += 1 # keeps track of sentence numbers
                j = 0  # keeps track of words within a sentence
                op_file.write("\n")
                continue
            word = line.strip() # remove any whitespace chars
            tag = predicted_tags[i][j]
            j += 1
            op_file.write(word+"\t"+tag+"\n")
    logging.info(f'tagged-file written to {tagged_file}')

if __name__ == '__main__':
    train_corpus = read_corpus()