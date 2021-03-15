import argparse
import numpy as np
import pickle
import itertools
from word2vec import load_pkl_obj, write_obj2file
from collections import Counter
from pathlib import Path
import logging

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


def fetch_aligned_pairs(src_file, trg_file, alignment_file):
    alignment_count = Counter()
    aligned_trg_words = set()
    aligned_src_words = set()

    with open(alignment_file) as af, open(src_file) as sf, open(trg_file) as tf:
        num = 1
        for alignments, src_line, trg_line in zip(af, sf, tf):
            word_alignments = alignments.split()
            src_words = src_line.split()
            trg_words = trg_line.split()

            for pair in word_alignments:
                try:
                    src_idx, trg_idx = pair.split('-')  # src-trg
                    alignment_count[(src_words[int(src_idx)], trg_words[int(trg_idx)])] += 1
                    aligned_trg_words.add(trg_words[int(trg_idx)])
                    aligned_src_words.add(src_words[int(src_idx)])
                except IndexError:
                    print(f"sent num: {num}")
                    print(f"src_idx {src_idx} trg_idx {trg_idx}\nsrc_words : {src_words}\ntrg_words: {trg_words}")
                    exit(0)
            num+=1

    logging.info(f'unique target words that are aligned : {len(aligned_trg_words)}')
    logging.info(f'unique src words that are aligned : {len(aligned_src_words)}')

    # -- save the alignment counter object for later use --
    # with open('aligned_output/saved_objects/src-trg-alignmentCount.pkl', 'wb') as out_file:
    #     pickle.dump(alignment_count, out_file)
    #
    # logging.info("saved alignment-count dictionary to aligned_output/saved_objects/src-trg-alignmentCount.pkl")

    return alignment_count, sorted(list(aligned_src_words)), sorted(list(aligned_trg_words))

def map_word2id(wordlist):
    word2id = dict()
    for i, word in enumerate(wordlist):
        word2id[word] = i
    return word2id


def create_alignment_matrix(alignment_count, src_word2id, trg_word2id):
    alignment_matrix = np.zeros((len(src_word2id), len(trg_word2id)), dtype=np.int8)

    for pair, count in alignment_count.items():
        src_word, trg_word = pair
        alignment_matrix[src_word2id[src_word], trg_word2id[trg_word]] = count
    np.save("aligned_output/saved_objects/alignment_matrix.npy", alignment_matrix)
    return alignment_matrix


def run_word_alignment_pipeline(src_file, trg_file, alignment_file):

    Path("./aligned_output/saved_objects/").mkdir(parents=True, exist_ok=True)
    alignment_count, src_wordVocab, trg_wordVocab = fetch_aligned_pairs(src_file, trg_file, alignment_file)

    # map src_wordVocab and trg_wordVocab to word IDs
    src_word2id = map_word2id(src_wordVocab)
    trg_word2id = map_word2id(trg_wordVocab)

    # --- save src, trg word vocab and word2id mappings to pkl object for later use ---
    write_obj2file(src_wordVocab, 'aligned_output/saved_objects/en-src-vocab.pkl')
    write_obj2file(trg_wordVocab, 'aligned_output/saved_objects/de-trg-vocab.pkl')
    write_obj2file(src_word2id, 'aligned_output/saved_objects/src-word2id.pkl')
    write_obj2file(trg_word2id, 'aligned_output/saved_objects/trg_word2id.pkl')
    logging.info("saved src, trg word vocab and word2id mappings to aligned_output/saved_objects/ as pkl objects..")

    # create alignment matrix
    alignment_matrix = create_alignment_matrix(alignment_count, src_word2id, trg_word2id)
    logging.info(f"alignment matrix shape: {alignment_matrix.shape}")


if __name__ == '__main__':
    src_file = './data/parallel_corpus/processed_en.src'
    trg_file = './data/parallel_corpus/processed_de.trg'
    alignment_file = './aligned_output/en-de-fullCorpus.align'


    run_word_alignment_pipeline(src_file, trg_file, alignment_file)