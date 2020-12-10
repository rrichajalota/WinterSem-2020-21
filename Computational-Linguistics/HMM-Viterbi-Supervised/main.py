#!/usr/bin/python3
"""
This script runs the entire pipeline from loading the corpus to training it, and running the viterbi-tagger.
"""

import os
import argparse
from time import time
from helper import *
from viterbi import viterbi
from HMM_training import perform_sanity_check, HMM_trainer
import matplotlib.font_manager
import matplotlib.pyplot as plt
import logging

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


def arg_parser():
    """ provides the user with options to train/test the HMM with different configurations. """
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help='use this option to provide a file (in CONLL format with words and POS-tags as columns) for training HMM')
    parser.add_argument("--test", help="use this option to provide a file (in CONLL format with words and POS-tags as columns) for testing the tagger")
    parser.add_argument("--plot_speed_curve", help="If specified, speed vs sentence-length curve is returned.", action="store_true")

    args = parser.parse_args()
    return args

def main():
    """
    runs the entire pipeline from loading the corpus to predicting the tagged output file.
    :return:
    """
    args = arg_parser()
    if args.train:
        train_corpus = read_corpus(path=args.train)
    else:
        train_corpus = read_corpus()

    if args.test:
        eval_corpus = read_corpus(path=args.test)
    else:
        eval_corpus = read_corpus(path='de-utb/de-eval.tt')

    start = time()
    # -- read the states and vocab from corpus --
    states, state_index = record_states_from_corpus(train_corpus)
    obs, obs_index = record_obs_from_corpus(train_corpus)

    # -- train HMM --
    initial_prob, transition_prob, emission_prob = HMM_trainer(train_corpus, states, state_index, obs, obs_index)
    perform_sanity_check(initial_prob, transition_prob, emission_prob) # checks if all probabilities add to 1.

    # -- call tagger on test corpus --
    eval_start = time()
    predicted_tags = []
    num = 0
    speed, sentence_length = [], []  # variables to plot speed vs sentence-length curve
    for sent in eval_corpus.tagged_sents():
        num += 1
        words = [tup[0] for tup in sent]
        best_path, elapsed_time = viterbi(states, words, initial_prob, transition_prob, emission_prob, obs_index)
        predicted_tags.append(best_path)
        speed.append(elapsed_time * 1000) # to convert in millisecs
        sentence_length.append(len(words))
    logging.info(f'Tagging the evaluation corpus containing {num} sentences takes {time()-start} secs.')

    print(max(sentence_length), max(speed))

    if args.plot_speed_curve:
        plt.plot(speed, sentence_length)
        plt.xlabel('time (in ms).')
        plt.ylabel('no. of words in a sentence')
        plt.title('Speed vs Sentence Length plot')
        plt.show()

    # -- write the tags to file --
    write_tags_to_file(predicted_tags=predicted_tags)
    logging.info(f'Time taken to run the entire pipeline: {time()-start} secs.')


if __name__ == '__main__':
    main()

