import argparse
import numpy as np
import itertools
from collections import Counter
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
                    trg_idx, src_idx = pair.split('-')  # trg-src
                    alignment_count[(src_words[int(src_idx)], trg_words[int(trg_idx)])] += 1
                    aligned_trg_words.add(trg_words[int(trg_idx)])
                    aligned_src_words.add(src_words[int(src_idx)])
                except IndexError:
                    print(num)
                    print(f"src_idx {src_idx} trg_idx {trg_idx}\nsrc_words : {src_words}\ntrg_words: {trg_words}")
                    exit(0)
            num+=1

    logging.info(f'unique target words that are aligned : {len(aligned_trg_words)}')
    logging.info(f'unique src words that are aligned : {len(aligned_src_words)}')
    return alignment_count, list(aligned_src_words), list(aligned_trg_words)

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
    np.save("aligned_output/alignment_matrix.npy", alignment_matrix)
    return alignment_matrix

# # for nnia
# def create_smaller_dataset(file):
#     with open(file) as f:
#         with open("data/de-test.tsv", 'w') as of:
#             count = 0
#             for line in f:
#                 words = line.split()
#                 of.write(f"{count}\t{words[0]}\t{words[1]}\n")
#                 count+=1
#                 if words[0] == ".":
#                     count = 0






if __name__ == '__main__':
    src_file = './data/parallel_corpus/processed_en.src'
    trg_file = './data/parallel_corpus/processed_de.trg'

    alignment_file = './aligned_output/mgiza.a'
    alignment_count, src_wordVocab, trg_wordVocab = fetch_aligned_pairs(src_file, trg_file, alignment_file)

    # map src_wordVocab and trg_wordVocab to word IDs
    src_word2id = map_word2id(src_wordVocab)
    trg_word2id = map_word2id(trg_wordVocab)

    #create alignment matrix
    alignment_matrix = create_alignment_matrix(alignment_count, src_word2id, trg_word2id)
    logging.info(f"alignment matrix shape: {alignment_matrix.shape}")