import numpy as np
import pickle as pkl
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from word2vec import *
from pathlib import Path
from pprint import pprint
import random


import argparse

def plot_loss_vs_epoch(de_epoch_loss, en_epoch_loss, src_vocab_size, trg_vocab_size):
    # Create count of the number of epochs
    epoch_count = range(len(de_epoch_loss))

    # Visualize training loss history
    fig = plt.figure()
    plt.plot(epoch_count, de_epoch_loss, color="red")
    plt.plot(epoch_count, en_epoch_loss, color="blue")
    plt.legend(['de', 'en'])
    plt.xlabel('Training Epoch')
    plt.ylabel('Loss')
    plt.title(f'en_vocab_size {src_vocab_size} de_vocab_size {trg_vocab_size}')
    fig.savefig('./results/visualizations/loss_vs_epoch.png', dpi=fig.dpi)


# Input vector, returns nearest word(s)
def cosine_similarity(word, weight_matrix, word_to_index, word_vocab):
    if type(word) == type('str'):
        # Get the index of the word from the dictionary
        index = word_to_index[word]
        # Get the correspondin weights for the word
        word_vector_1 = weight_matrix[index]
    else:
        word_vector_1 = word

    word_similarity = {}

    for i in range(len(word_vocab)):
        word_vector_2 = weight_matrix[i]

        theta = compute_similarity_between(word_vector_1, word_vector_2)

        word = word_vocab[i]
        word_similarity[word] = theta

    return word_similarity


def compute_similarity_between(word_vector_1, word_vector_2):
    theta_sum = np.dot(word_vector_1, word_vector_2)
    theta_den = np.linalg.norm(word_vector_1) * np.linalg.norm(word_vector_2)
    theta = theta_sum / theta_den
    return theta


def fetch_similar_words(top_n_words, words_subset, weight_matrix, word_to_index, word_vocab):
    sim_words = {}

    for word in words_subset:

        # Get the similar words (dict: {word, similarity_score}) for the word: word
        similar_words = cosine_similarity(word, weight_matrix, word_to_index, word_vocab)

        # Sort the top_n_words
        words_sorted = sorted(similar_words.items(), key=lambda kv: kv[1], reverse=True)[1:top_n_words+1]

        sim_words[word] = words_sorted

    for wd, sim_wds in sim_words.items():
        pprint(f"{wd} : {str(sim_wds)}")


def plot_tsne_for_word_similarity(weight_matrix, word_to_index, word_vocab, lang, sample=500, seed=30, perplexity=15,
                                  n_iterations=5000, learning_rate=40, lang1_vocab_ratio=0.5):
    """
    Recommneded values as per the sample size: Sample=250, perplexity=15, n_iter=5000, learning_rate=40
    --sample 200 --lr 35 --perplexity 14 --seed 30 (to reproduce the plot in results)
    ---------
   Parameters
   ----------
   lang - 'en', 'de', 'en-de', 'de-en', etc. Note: For bilingual plots, make sure to seperate the 2 languages by a '-'!
   lang1_vocab_ratio - In case of bilingual embeddings, if len(vocab_lang1) > len(vocab_lang2), set a ratio for lang1-lang2 i.e.
                        value between [0,1] for it. default = 0.5 i.e. both languages are equally represented.
    """
    labels = []
    tokens = []

    random.seed(seed)
    if '-' in lang: # means bilingual
        n = int(sample/2)
        lang1_size = int(len(word_vocab)*lang1_vocab_ratio)
        rand_ints = random.sample(range(0,lang1_size), n) + random.sample(range(lang1_size, len(word_vocab)), n)
    else:
        rand_ints = random.sample(range(0,len(word_vocab)), sample) # random.sample() ensures unique values


    # create t-sne plot for random n samples of the vocab
    for idx in rand_ints:
        tokens.append(weight_matrix[word_to_index[word_vocab[idx]]])
        labels.append(word_vocab[idx])

    assert len(labels) == len(set(labels))

    # TSNE : Compressing the weights to 2 dimensions to plot the data, 
    tsne_model = TSNE(perplexity=perplexity, n_components=2, init='pca', n_iter=n_iterations, random_state=seed, learning_rate=learning_rate)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i],
                      xy=(x[i], y[i]),
                      xytext=(5, 2),
                      textcoords='offset points',
                      ha='right',
                      va='bottom')
    # plt.title(plot_title)
    plt.savefig('./results/visualizations/' + lang + str(sample) + '_t-sne_plot_.png', dpi=300)
    plt.show()



def main():
    args = arg_parser()
    assert args.lang1_ratio >= 0 and args.lang1_ratio <= 1

    Path("./results/visualizations/").mkdir(parents=True, exist_ok=True)

    wd2id = load_pkl_obj(args.word_to_index_file)
    vocab = load_pkl_obj(args.word_vocab_file)
    wts = np.load(args.wt_matrix_file, allow_pickle=True)

    print(f"emb_matrix shape: {wts.shape} vocab len: {len(vocab)} word2id len: {len(wd2id)}")

    plot_tsne_for_word_similarity(wts, wd2id, vocab, lang=args.lang, sample=args.sample, seed=args.seed, perplexity=args.perplexity,
                                  n_iterations=args.n_iter, learning_rate=args.lr, lang1_vocab_ratio=args.lang1_ratio)
    #
    # if args.de_loss is not None and args.en_loss is not None:
    #     plot_loss_vs_epoch(args.de_loss, args.en_loss)


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("wt_matrix_file", help='use this option to provide the path for embedding weights')
    parser.add_argument("word_to_index_file",
                        help="path for word2index file")
    parser.add_argument("word_vocab_file",
                        help="path for word_vocab file")
    parser.add_argument("lang",
                        help="specify the lang. abbreviation. for bilingual use, en-de format")
    parser.add_argument("--sample", default=100, type=int,
                        help="number of data points to be considered for T-SNE. (default 100)")
    parser.add_argument("--seed", default=30, type=int,
                        help="random seed for T-SNE and for sampling random values from the weight matrix. (default 30)")
    parser.add_argument("--perplexity", default=15, type=int,
                        help="perplexity value foor T-SNE (5-50). For N samples, keep perplexity ~ sqrt(N). (default 15)")
    parser.add_argument("--n_iter", default=5000, type=int,
                        help="number of iterations for training T-SNE. (default 5000)")
    parser.add_argument("--lr", default=40, type=int,
                        help="learning rate for T-SNE. (default 40)")
    parser.add_argument("--lang1_ratio", default=0.5, type=float,
                        help="In case of bilingual embeddings, if len(vocab_lang1) > len(vocab_lang2), "
                             "set a ratio for lang1-lang2 i.e. value between [0,1] for it. default = 0.5 i.e. "
                             "both languages are equally represented (default 0.5)")
    # parser.add_argument("--de_loss",
    #                     help="path for loss_vs_epoch object to plot the loss vs epoch plot. also "
    #                          "input the loss_per_epoch object for en. Both losses will be plotted on the same graph")
    # parser.add_argument("--en_loss",
    #                     help="path for loss_vs_epoch object to plot the loss vs epoch plot. also "
    #                          "input the loss_per_epoch object for de. Both losses will be plotted on the same graph")
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    main()
    # --- uncomment the following lines until line 202 to get epoch vs loss plot ---
    # src_emb_vocab_path = "data/temp_files/en/word_vocab.pkl"
    # trg_emb_vocab_path = "data/temp_files/de/word_vocab.pkl"
    # #
    # src_emb_wd2id_path = "data/temp_files/en/word2id.pkl"
    # trg_emb_wd2id_path = "data/temp_files/de/word2id.pkl"
    #
    # Path("./results/visualizations/").mkdir(parents=True, exist_ok=True)
    #
    # src_emb_vocab = load_pkl_obj(src_emb_vocab_path)
    # trg_emb_vocab = load_pkl_obj(trg_emb_vocab_path)
    #
    # de_epoch_loss = load_pkl_obj("./data/temp_files/de/epoch_loss.pkl")
    # en_epoch_loss = load_pkl_obj("./data/temp_files/en/epoch_loss.pkl")
    # logging.info(f"de epoch loss {de_epoch_loss} en epoch loss {en_epoch_loss}")
    #
    # plot_loss_vs_epoch(de_epoch_loss, en_epoch_loss, len(src_emb_vocab), len(trg_emb_vocab))
    #
    # src_wt_mat = np.load("data/temp_files/en/w1.npy", allow_pickle=True)
    # trg_wt_mat = np.load("data/temp_files/de/w1.npy", allow_pickle=True)
    #


    #print(src_vocab)
    #print("\n\n")
    #print(trg_emb_vocab)

    #
    # src_emb_wd2id = load_pkl_obj(src_emb_wd2id_path)
    # trg_emb_wd2id = load_pkl_obj(trg_emb_wd2id_path)

    # bilingual_wd2id = load_pkl_obj("data/temp_files/bilingual-word2id.pkl")
    # bilingual_vocab = load_pkl_obj("data/temp_files/bilingual-wordVocab.pkl")
    # bilingual_wts = np.load("data/temp_files/bilingual_emb_de-en.npy", allow_pickle=True)

    # print(f"{bilingual_wts.shape} vocab {len(bilingual_vocab)} word2id {len(bilingual_wd2id)}")


    #logging.info(f"src emb vocab {src_vocab}\n trg emb vocab {trg_emb_vocab}")

    # de_words = ["zahl", 'zentrum', 'zerschlag', 'zerschnitt', 'wunschlist', 'wunscht', 'wurd', 'wohn', 'wissensaustausch',
    #             'wissenschaft', 'wieso', 'wieviel', 'typisch', 'sozialbereich', 'sommerurlaub', 'sieht', 'sicher', 'ruhr',
    #             'sauber', 'sag']
    #
    # en_words = ['guilt', 'heaven', 'high', 'grade', 'government', 'goal', 'god', 'hire', 'farm', 'food', 'empress',
    #             'effect', 'drug', 'ecologist', 'dutch', 'edinburgh', 'consult', 'construct', 'cost', 'critic', 'bulk', 'bank']
    #
    # fetch_similar_words(top_n_words=5, words_subset=de_words, weight_matrix=trg_wt_mat, word_to_index=trg_emb_wd2id, word_vocab=trg_emb_vocab)
    #
    # fetch_similar_words(top_n_words=5, words_subset=en_words, weight_matrix=src_wt_mat, word_to_index=src_emb_wd2id, word_vocab=src_vocab)

    #plot_tsne_for_word_similarity(src_wt_mat, src_emb_wd2id, src_vocab, lang='en', sample=300)

    #plot_tsne_for_word_similarity(trg_wt_mat, trg_emb_wd2id, trg_emb_vocab, lang='de', sample=300)

    #plot_tsne_for_word_similarity(bilingual_wts, bilingual_wd2id, bilingual_vocab, lang='de-en', sample=200)