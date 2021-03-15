import pickle as pkl
import numpy as np
from analysis import *
from scipy.stats.stats import pearsonr


def squish_range(val, low=0, high=5):
    """
    squishes range to (0, 1)
    https://stats.stackexchange.com/questions/70801/how-to-normalize-data-to-0-1-range
    """
    # newvalue = a * value + b. a = (max'-min')/(max-min) and b = max' - a * max
    a = 1 / (high-low)
    b = 1 - (a * high)
    return (a * val + b)



def word_similarity_task(ws_test_file, weights, word2id, low=0, high=1):
    """
    The higher the scores, the better the embeddings.
    """
    true_scores = []
    pred_scores = []
    oov_pairs = []
    total = 0
    with open(ws_test_file) as f:
        for line in f:
            total +=1
            src_wd, trg_wd, gold_score = line.split()
            gold_score = float(gold_score)
            if low < 0 or high > 1:
                gold_score = squish_range(gold_score, low, high)

            if {src_wd, trg_wd} <= word2id.keys():
                #print(src_wd, trg_wd)
                pred_score = compute_similarity_between(weights[word2id[src_wd]], weights[word2id[trg_wd]])
                # cosine_sim value [-1, 1] scale it to range [0,1]
                # # newvalue = a * value + b. a = (max'-min')/(max-min) and b = max' - a * max
                # https://stats.stackexchange.com/questions/70801/how-to-normalize-data-to-0-1-range
                scaled_pred_score = 0.5 * pred_score + 0.5
                true_scores.append(gold_score)
                pred_scores.append(scaled_pred_score)
            else:
                oov_pairs.append(line)
    #print(pred_scores, true_scores)
    print("-- Semantic Similarity task --")
    if len(pred_scores) >= 2:
        print(f"Pearson Correlation score: {pearsonr(true_scores, pred_scores)}")
    #print(f"total pairs: {total} oov pairs: {len(oov_pairs)}")
    print(f"oov pairs: {len(oov_pairs)} out of {total}")


def analogy_task(eval_file, weights, word2id, precision):
    total, oov, correct = 0, 0, 0

    with open(eval_file) as f:
        for line in f:
            a, b, c, d = line.split()
            if {a, b, c} <= word2id.keys():
                vec_a, vec_b, vec_c = weights[word2id[a]], weights[word2id[b]], weights[word2id[c]]
                vec = vec_b - vec_a + vec_c
                similar_words = cosine_similarity(vec, weights, word2id, list(word2id.keys())) # dict of words
                # Sort the dict to get top k similar words - precision @ k
                words_sorted = {k: v for k, v in sorted(similar_words.items(), key=lambda item: item[1], reverse=True)[:precision]}
                logging.info(line, "\t", words_sorted)
                total += 1
                if d in words_sorted.keys():
                    correct += 1
            else:
                print(a, b, c, d)
                oov += 1
    print(f"Analogy Correct predictions: {correct} out of {total}. {oov} oov pairs not considered.")

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--word_sim", default="data/eval/custom_en-de-similarity.txt",
                        help='use this option to provide the path for word similarity evaluation file')
    parser.add_argument("--analogy", default="data/eval/cutsom_de_en_analogy.txt",
                        help="use this option to provide the path for analogy test file")
    parser.add_argument("--weights", default="data/temp_files/bilingual_1000/bilingual_emb_de-en.npy",
                        help="use this option to provide the path for embedding weights.")
    parser.add_argument("--wd2id", default="data/temp_files/bilingual_1000/bilingual-word2id.pkl",
                        help="use this option to provide the path for the word2id dict object for the embeddings.")
    parser.add_argument("--min", default=0, type=int,
                        help="use this option to provide the minimum value of the range used for human rating in word similarity eval file")
    parser.add_argument("--max", default=1, type=int,
                        help="use this option to provide the maximum value of the range used for human rating in word similarity eval file")

    args = parser.parse_args()
    return args



# evaluate monolingual and bilingual embeddings
if __name__ == '__main__':
    args = arg_parser()
    bilingual_wd2id = load_pkl_obj(args.wd2id)
    bilingual_wts = np.load(args.weights, allow_pickle=True)

    word_similarity_task(args.word_sim, bilingual_wts, bilingual_wd2id, low=args.min, high=args.max)
    print("-- Analogy Task --")
    analogy_task(args.analogy,bilingual_wts, bilingual_wd2id, precision=7)

    # src_emb_vocab_path = "data/temp_files/en/word_vocab.pkl"
    # trg_emb_vocab_path = "data/temp_files/de/word_vocab.pkl"
    #
    # src_emb_wd2id_path = "data/temp_files/en/word2id.pkl"
    # trg_emb_wd2id_path = "data/temp_files/de/word2id.pkl"
    #
    # src_wt_mat = np.load("data/temp_files/en/w1.npy", allow_pickle=True)
    # trg_wt_mat = np.load("data/temp_files/de/w1.npy", allow_pickle=True)


