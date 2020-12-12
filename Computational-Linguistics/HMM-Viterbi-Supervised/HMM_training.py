
import numpy as np
from helper import *
from time import time
import logging

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


def perform_sanity_check(initial_prob, transition_prob, emission_prob):
    '''
    This function makes sure all the probabilities sum up to 1.
    :param initial_prob:
    :param transition_prob:
    :param emission_prob:
    :return: throws an assertion error if any of the assertion statement is untrue, else nothing.
    '''
    assert (np.sum(initial_prob) == 1)
    assert (all(np.sum(transition_prob, axis=1)) == 1)
    assert (all(np.sum(emission_prob, axis=1)) == 1)


def HMM_trainer(train_corpus, states, state_index, vocab, vocab_index, smoothing_factor=None, word_handling=False):
    '''
    :param train_corpus: of type nltk.corpus.reader.conll.ConllCorpusReader
    :param states: unique list of POS-tags from the corpus
    :param state_index: state2index dictionary e.g. {'NN': 0, 'PN': 4,...}
    :param vocab: list of unique words found in the training corpus
    :param vocab_index: vocab dict containing observed words in training mapped to an index {'der': 12, 'schon': 0,...}
    :param smoothing_factor: for add-k smoothing. k = [0.1, 1.0]. Note: k = 1 is equivalent to laplace smoothing.
    :return: estimated initial, transition and emission probability matrices
    '''
    start = time()
    initial_prob = np.zeros(len(states))  # initial probab vector for all states - len N - initialized with 0s
    transition_prob = np.zeros((len(states), len(states)))  # NxN zero-matrix -> to contain P( X_t+1 = s_j | X_t = s_i)
    emission_prob = np.zeros((len(states), len(vocab) + 1)) if word_handling else np.zeros((len(states), len(vocab))) # Nx(T+1) zero-matrix -> to contain P(Y_t = o | X_t = s_i ) extra column to account for unknown word

    for sent in train_corpus.tagged_sents():
        prev_state_index = -1

        for i, tup in enumerate(sent): # tup: ('der', DET)
            hidden_state_index = state_index[tup[1]]
            obs_col = vocab_index[tup[0]] if tup[0] in vocab_index else len(vocab) # last col. considered as <unk>

            if i == 0:  # first word of the sentence
                initial_prob[hidden_state_index] += 1
            else:
                transition_prob[prev_state_index][hidden_state_index] += 1

            emission_prob[hidden_state_index][obs_col] += 1
            prev_state_index = hidden_state_index

    # divide the count by the corresponding totals to get probability
    initial_prob /= len(train_corpus.tagged_sents())
    transition_prob /= np.sum(transition_prob, axis=1).reshape(-1, 1)
    #print(transition_prob.shape)

    if smoothing_factor: # add 1 to all counts in emission_prob matrix
        emission_prob = perform_addK_smoothing(emission_prob, vocab, k=smoothing_factor)
    else:
        emission_prob /= np.sum(emission_prob, axis=1).reshape(-1, 1)

    logging.info(f'HMM training takes {time() - start} secs.')

    return (initial_prob, transition_prob, emission_prob)


def perform_addK_smoothing(emission_prob, vocab, k=1): # laplace by default
    emission_prob = np.add(emission_prob, k)
    emission_prob /= (np.sum(emission_prob, axis=1).reshape(-1, 1) + k * len(vocab))
    return emission_prob


if __name__ == '__main__':
    train_corpus = read_corpus()
    states, state_index = record_states_from_corpus(train_corpus)
    vocab, vocab_index = extract_vocab_from_corpus(train_corpus)

    initial_prob, transition_prob, emission_prob = HMM_trainer(train_corpus, states, state_index, vocab, vocab_index)
    perform_sanity_check(initial_prob, transition_prob, emission_prob)


