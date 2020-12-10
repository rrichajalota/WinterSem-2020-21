
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


def HMM_trainer(train_corpus, states, state_index, obs, obs_index):
    '''
    :param train_corpus: of type nltk.corpus.reader.conll.ConllCorpusReader
    :param states: unique list of POS-tags from the corpus
    :param state_index: state2index dictionary e.g. {'NN': 0, 'PN': 4,...}
    :param obs: list of unique words found in the training corpus
    :param obs_index: vocab dict containing observed words in training mapped to an index {'der': 12, 'schon': 0,...}
    :return: estimated initial, transition and emission probability matrices
    '''
    start = time()
    initial_prob = np.zeros(len(states))  # initial probab vector for all states - len N - initialized with 0s
    emission_prob = np.zeros((len(states), len(obs)))  # NxT zero-matrix -> to contain P(Y_t = o | X_t = s_i )
    transition_prob = np.zeros((len(states), len(states)))  # NxN zero-matrix -> to contain P( X_t+1 = s_j | X_t = s_i)

    for sent in train_corpus.tagged_sents():
        prev_state_index = -1

        for i, tup in enumerate(sent):
            hidden_state_index = state_index[tup[1]]
            obs_col = obs_index[tup[0]]

            if i == 0:  # first word of the sentence
                initial_prob[hidden_state_index] += 1
            else:
                transition_prob[prev_state_index][hidden_state_index] += 1

            emission_prob[hidden_state_index][obs_col] += 1
            prev_state_index = hidden_state_index

    # divide the count by the corresponding totals to get probability
    initial_prob /= len(train_corpus.tagged_sents())
    transition_prob /= np.sum(transition_prob, axis=1).reshape(-1, 1)
    emission_prob /= np.sum(emission_prob, axis=1).reshape(-1, 1)

    logging.info(f'HMM training takes {time()-start} secs.')

    return (initial_prob, transition_prob, emission_prob)


if __name__ == '__main__':
    train_corpus = read_corpus()
    states, state_index = record_states_from_corpus(train_corpus)
    obs, obs_index = record_obs_from_corpus(train_corpus)

    initial_prob, transition_prob, emission_prob = HMM_trainer(train_corpus, states, state_index, obs, obs_index)
    perform_sanity_check(initial_prob, transition_prob, emission_prob)


