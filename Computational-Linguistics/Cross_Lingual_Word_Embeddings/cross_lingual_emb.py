import numpy as np
import pickle as pkl
import logging
from time import time
from word2vec import *

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)



def learn_linear_transformation():
    # method without word alignments - linear projection approach by Mikolov et al.
    pass


def project_source2target(alignment_matrix, trg_wt_mat, trg_vocab, src_vocab, src_en_wd2id, trg_emb_wd2id):
    src_translation_emb = []
    src_translation_wd2id = {}
    id = 0
    start = time()
    src_translation_word_vocab = []

    # loop over the src embeddings because len(src_embeddings) < len(src_vocab_alignments)
    for i in range(len(src_vocab)):
        logging.info(i)
        # fetch the word to be translated
        src_word = src_vocab[i]

        # find this word in the alignment matrix src-trg
        if src_word not in src_en_wd2id:
            #if fastAlign does not find an alignment for a src word, the src word would not exist in the vocab
           continue
        src_idx_am = src_en_wd2id[src_word]

        # initialize the translated_embedding vector for this SRC word
        src_word_translated_vec = [0] * trg_wt_mat.shape[1]
        #logging.info(f"src_word_translated_vec {len(src_word_translated_vec)} trg_wt_mat {trg_wt_mat.shape}")

        # fetch the row for src_idx from the alignment matrix, am
        src_idx_row = alignment_matrix[src_idx_am]
        oov_emb_count = 0
        st = time()

        # use the eqn given in Guo, J., Che, W., Yarowsky, D., Wang, H., & Liu, T. (2015).
        # Cross-lingual Dependency Parsing Based on Distributed Representations.
        # to get translation for src_word
        for j, count in enumerate(src_idx_row):
            try:
                # if there is alignment between i-th src word and j-th trg word
                if count != 0:
                    # fetch the target word
                    aligned_trg_wd = trg_vocab[j]

                    # find the idx of this word in the embedding matrix
                    if aligned_trg_wd in trg_emb_wd2id.keys():
                        trg_wd_idx_em = trg_emb_wd2id[aligned_trg_wd]
                        trg_wd_emb = trg_wt_mat[trg_wd_idx_em]
                        src_word_translated_vec += count * trg_wd_emb

                    else: # if there is an alignment but embedding doesn't exist
                        oov_emb_count += count # to later subtract from np.sum(). these alignments would be
                        # considered 0 for fair normalization
            except IndexError:
                logging.warning(f"Indexing error!!! with {trg_vocab[j]}\t trg_emb_wd2id: {trg_emb_wd2id[trg_vocab[j]]} ")
                break

        if all(v == 0 for v in src_word_translated_vec):
            # no alignment with any of the target words (would happen if trg word embedding is missing)
            continue

        # normalize the vector - i.e. take the avg.
        src_word_translated_vec /= (np.sum(src_idx_row) - oov_emb_count)
        # append the translated src_word embedding to the src_translation_emb matrix
        src_translation_emb.append(src_word_translated_vec)

        src_translation_word_vocab.append(src_word)
        src_translation_wd2id[src_word] = id
        id += 1

        logging.info(f"time taken to loop over the target word alignments {time()- st}")

    logging.info(f"time taken to loop over all the src indexes: {time()- start}")

    return np.asarray(src_translation_emb), src_translation_word_vocab, src_translation_wd2id # shape (len(src_vocab), trg_wt_mat.shape()[1])


def generate_bilingual_embeddings(src_translation_emb, src_translation_word_vocab, src_translation_wd2id,
                                  trg_wt_mat, trg_emb_vocab, trg_emb_wd2id, unique= True):
    """

    Parameters
    ----------
    src_translation_emb - embeddings for SRC words in the TRG vector space
    src_translation_word_vocab
    src_translation_wd2id
    trg_wt_mat - TRG word embeddings
    trg_emb_vocab
    trg_emb_wd2id
    unique - if words in src_translation_word_vocab exist in trg_emb_vocab, there are 2 options for representing such
    words in the common vector space. 1) (unique=True) average out the word embeddings for such words and keep only unique representations.
    2) (unique=False)keep non-unique representations as they might depict different word senses, rot-de vs rot-en, will-de vs will-en

    """

    if unique: ## OPT1: avg the embeddings
        common_words = set(set(trg_emb_vocab) & set(src_translation_word_vocab))
        uniq_src_translated_words = [] # new word vocab
        uniq_wd2id = {}

        if len(common_words) > 0:
            logging.info("common words found. Averaging the embeddings for these words..")

        for word in common_words:
            trg_wt_mat[trg_emb_wd2id[word]] = (trg_wt_mat[trg_emb_wd2id[word]] + src_translation_emb[
                src_translation_wd2id[word]]) / 2

        # append the remaining values to the trg matrix
        for word in src_translation_word_vocab:
            if word not in common_words:
                trg_wt_mat = np.vstack((trg_wt_mat, src_translation_emb[src_translation_wd2id[word]]))
                uniq_src_translated_words.append(word)

        # create word2id mapping
        trg_emb_vocab_size = len(trg_emb_vocab)
        for i, word in enumerate(uniq_src_translated_words):
            uniq_wd2id[word] = i + trg_emb_vocab_size

        bilingual_trg_src_wd2id = dict(trg_emb_wd2id, **uniq_wd2id)
        logging.info(f"bilingual_trg_src_wd2id {len(bilingual_trg_src_wd2id)}, len(uniq_translated_src_wd2id) "
                     f"{len(uniq_wd2id)} + trg_emb_vocab_size {trg_emb_vocab_size} ")
        #assert len(bilingual_trg_src_wd2id) == len(uniq_wd2id) + trg_emb_vocab_size

        # bilingual word-vocab
        bilingual_vocab = trg_emb_vocab + uniq_src_translated_words

        # save objects
        write_obj2file(bilingual_vocab, "data/temp_files/bilingual_1000/bilingual-wordVocab.pkl")
        write_obj2file(bilingual_trg_src_wd2id, "data/temp_files/bilingual_1000/bilingual-word2id.pkl")
        np.save("data/temp_files/bilingual_1000/bilingual_emb_de-en.npy", trg_wt_mat)

    # else: ## --- TODO: evaluate this approach ---
    #     # OPT2: concatenate the trg and src-translated embeddings row-wise, keeping duplicates
    #     bilingual_trg_src_concat_emb = np.concatenate((trg_wt_mat, src_translation_emb), axis=0)
    #
    #     # offset the word2id mapping for src_translation by len(trg_emb_vocab) and create a joint word2id mapping
    #     trg_emb_vocab_size = len(trg_emb_vocab)
    #     for k in src_translation_wd2id:
    #         src_translation_wd2id[k] += trg_emb_vocab_size
    #
    #     #for words that are the same in 2 languages, 2 diff emb exist
    #     write_obj2file(src_translation_wd2id,
    #                    "data/temp_files/src_translation_wd2id.pkl")  # use both src_translation and trg_wd2id
    #
    #     # generate bilingual vocab
    #     concat_emb_vocab = trg_emb_vocab + src_translation_word_vocab  # not-unique
    #
    #     # save weights and vocab files
    #     # write_obj2file(bilingual_trg_src_wd2id, "data/temp_files/bilingual-wd2id.pkl")
    #     write_obj2file(concat_emb_vocab, "data/temp_files/bilingual-wordVocab.pkl")
    #
    #     np.save("data/temp_files/bilingual_emb_de-en_concatenated.npy", bilingual_trg_src_concat_emb)


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--align_mat", default="aligned_output/saved_objects/alignment_matrix.npy",
                        help='use this option to provide the path for npy object for alignment_matrix')
    parser.add_argument("--target_weights", default="data/temp_files/de/w1.npy",
                        help="use this option to provide the path for target weights (de) numpy object")
    parser.add_argument("--trg_vocab_wa", default="aligned_output/saved_objects/de-trg-vocab.pkl",
                        help="use this option to provide the path for the target vocabulary of word alignments corpus.")
    parser.add_argument("--src_vocab_wa", default="aligned_output/saved_objects/en-src-vocab.pkl",
                        help="use this option to provide the path for the source vocabulary of word alignments corpus.")
    parser.add_argument("--src_wd2id_wa", default="data/temp_files/en/word2id.pkl",
                        help="use this option to provide the path for the source word2id mapping of word alignments.")
    parser.add_argument("--trg_wd2id_emb", default="data/temp_files/de/word2id.pkl",
                        help="use this option to provide the path for the target word2id mapping of trained word embeddings.")
    parser.add_argument("--trg_vocab_emb", default="data/temp_files/de/word_vocab.pkl",
                        help="use this option to provide the path for the target word vocab of trained word embeddings.")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = arg_parser()
    # src_emb_vocab_path = "data/temp_files/en/word_vocab.pkl"
    # trg_emb_vocab_path = "data/temp_files/de/word_vocab.pkl"
    #
    # src_emb_wd2id_path = "data/temp_files/en/word2id.pkl"
    # trg_emb_wd2id_path = "data/temp_files/de/word2id.pkl"
    # de_trg_vocab_path = "aligned_output/saved_objects/de-trg-vocab.pkl"
    # en_src_vocab_path = "aligned_output/saved_objects/en-src-vocab.pkl"
    # src_en_wd2id_path = "aligned_output/saved_objects/src-word2id.pkl"
    # trg_de_wd2id_path = "aligned_output/saved_objects/trg_word2id.pkl"

    start = time()
    # load all saved objects
    alignment_matrix = np.load(args.align_mat, allow_pickle=True)
    src_wt_mat = np.load("data/temp_files/en/w1.npy", allow_pickle=True)
    trg_wt_mat = np.load(args.target_weights, allow_pickle=True)

    trg_vocab = load_pkl_obj(args.trg_vocab_wa)
    src_vocab = load_pkl_obj(args.src_vocab_wa)
    trg_emb_vocab = load_pkl_obj(args.trg_vocab_emb)

    src_en_wd2id = load_pkl_obj(args.src_wd2id_wa)
    trg_emb_wd2id = load_pkl_obj(args.trg_wd2id_emb)
    logging.info(f"loaded all pickle files. took {time()-start} secs.")

    src_translation_emb, src_translation_word_vocab, src_translation_wd2id = \
        project_source2target(alignment_matrix, trg_wt_mat, trg_vocab, src_vocab, src_en_wd2id, trg_emb_wd2id)

    generate_bilingual_embeddings(src_translation_emb, src_translation_word_vocab, src_translation_wd2id,
                                  trg_wt_mat, list(trg_emb_wd2id.keys()), trg_emb_wd2id, unique= True)















