#!/usr/bin/python3
"""
This script runs the entire pipeline from loading the corpus to training it, and running the viterbi-tagger.
"""

import os
import argparse
from time import time
from helper import *
from viterbi import viterbi
from HMM_training import *
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
    parser.add_argument("--tagged_file_path", help="provide filename or filepath for saving the (POS-)tagged output file.")
    parser.add_argument("--smoothing_factor", help="Provide a value between [0, 1] for add-k smoothing. By default, crude smoothing is done. " \
                                                   "Note: k = 1 is equivalent to laplace smoothing. Check READme for definitions and more.")
    parser.add_argument("--frequency_threshold", help="All words that occur less than the given integer value (in the training corpus) will be removed.")
    parser.add_argument("--set_vocab_size",
                        help="Either enter a value between 0 and 1, or a numerical value. Only the most commonly occuring words (=entered value) would be considered in the vocab.")

    args = parser.parse_args()
    return args


def main():
    """
    runs the entire pipeline from loading the corpus to predicting the tagged output file.
    :return:
    """
    args = arg_parser()

    train_corpus = read_corpus(path=args.train) if args.train else read_corpus()
    eval_corpus = read_corpus(path=args.test) if args.test else read_corpus(path='de-utb/de-eval.tt')

    start = time()
    # -- read the states and vocab from corpus --
    states, state_index = record_states_from_corpus(train_corpus)
    vocab, vocab_index = extract_vocab_from_corpus(train_corpus, frequency_threshold=args.frequency_threshold, vocab_size=args.set_vocab_size)

    word_handling = True if (args.frequency_threshold or (args.set_vocab_size and float(args.set_vocab_size) != 1.0)) else False

    # -- train HMM --
    initial_prob, transition_prob, emission_prob = HMM_trainer(train_corpus, states, state_index, vocab, vocab_index,
                                                               smoothing_factor=float(args.smoothing_factor), word_handling=word_handling)
    perform_sanity_check(initial_prob, transition_prob, emission_prob) # checks if all probabilities add to 1.

    # -- call tagger on test corpus --
    eval_start = time()
    predicted_tags = []
    num = 0
    speed, sentence_length = [], []  # variables to plot speed vs sentence-length curve
    for sent in eval_corpus.tagged_sents():
        num += 1
        words = [tup[0] for tup in sent]
        best_path, elapsed_time = viterbi(states, words, initial_prob, transition_prob, emission_prob, vocab_index, word_handling=word_handling)
        predicted_tags.append(best_path)
        speed.append(elapsed_time * 1000) # to convert in millisecs
        sentence_length.append(len(words))
    logging.info(f'Tagging the evaluation corpus containing {num} sentences takes {time()-eval_start} secs.')

    if args.plot_speed_curve:
        plt.plot(speed, sentence_length)
        plt.xlabel('time (in ms).', fontsize = 14)
        plt.ylabel('no. of words in a sentence', fontsize = 14)
        plt.title('Speed vs Sentence Length plot', fontsize = 14)
        plt.show()

    # -- write the tags to file --
    out_file = args.tagged_file_path if args.tagged_file_path else  "outputs/de-tagged.tt"
    write_tags_to_file(predicted_tags=predicted_tags, tagged_file=out_file)

    logging.info(f'Time taken to run the entire pipeline: {time()-start} secs.')


if __name__ == '__main__':
    main()

