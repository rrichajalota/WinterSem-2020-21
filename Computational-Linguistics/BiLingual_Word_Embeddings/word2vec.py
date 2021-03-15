import numpy as np
from scipy.sparse import csc_matrix
from collections import Counter
import logging
import re
import pickle
from pathlib import Path

import argparse



np.random.seed(42)


def load_pkl_obj(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def write_obj2file(obj, filename):
    with open(filename, 'wb') as f:
        return pickle.dump(obj, f)


class word2vec(object): # skipgram model
    def __init__(self, corpus, sample_size, window_size, lang, dims=100, n_epochs=5, lr= 0.01, dir="data/temp_files/",
                 generate_training_data=True, batch_size=500, min_word_freq=5):
        """

        Parameters
        ----------
        corpus - filepath
        dir - path to dir where all temp files will be stored
        sample_size - #samples to be taken from the corpus
        window_size - #context words to consider. choose a value between 2-5
        """
        self.dir = dir + lang + "/"
        self.lang = lang
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.dims = dims
        self.window_size = window_size
        self.corpus_path = corpus
        self.min_freq = min_word_freq
        # make sure self.dir path exists in the file system
        Path(self.dir).mkdir(parents=True, exist_ok=True)
        # save the subset of the corpus = sample_size to the temp folder
        self.create_smaller_corpus(sample_size)
        # build a vocabulary from the smaller corpus
        self.word_vocab = self.fetch_word_vocab()
        self.vocab_size = len(self.word_vocab)
        self.word2id = self.map_word2id()
        if generate_training_data is True:
            logging.info(f"Starting training data generation..")
            self.generate_training_data()

    def create_smaller_corpus(self, sample_size):
        """
        removes punctuations and digits from the corpus. saves the tokenized text to another file.

        """
        with open(self.corpus_path) as f:
            self.smaller_corpus_path = self.dir+ str(sample_size) + self.lang + "_corpus.txt"
            with open(self.smaller_corpus_path, "w") as tf:
                for i, line in enumerate(f):
                    if i == sample_size: break
                    tf.write(line.strip()+"\n")
        logging.info(f"smaller corpus saved to {self.smaller_corpus_path}")

    def map_word2id(self):
        word2id = dict()
        for i, word in enumerate(self.word_vocab):
            word2id[word] = i

        write_obj2file(word2id, self.dir+'word2id.pkl')
        logging.info(f"vocab size {self.vocab_size}")
        logging.info(f"Saved the word2id dict mapping to {self.dir}word2id.pkl")

        write_obj2file(self.word_vocab, self.dir+'word_vocab.pkl')
        logging.info(f"Saved the word_vocab to {self.dir}word_vocab.pkl")

        return word2id

    def fetch_word_vocab(self):
        word_freq = Counter()
        with open(self.smaller_corpus_path) as f:
            for line in f:
                for word in line.split():
                    word_freq[word] += 1
        logging.info(f"number of unique words: {len(word_freq)}")
        if self.min_freq != 0:
            return sorted([k for k, v in word_freq.items() if v >= self.min_freq ])

        return sorted(word_freq.keys())

    def get_one_hot_encoding(self, word):
        word_vec = np.zeros(self.vocab_size)
        word_idx = self.word2id[word]
        word_vec[word_idx] = 1
        return word_vec

    def generate_training_data(self):
        training_data = []
        batch = 1
        Path(self.dir+"batch_training").mkdir(parents=True, exist_ok=True)

        with open(self.smaller_corpus_path) as f:

            for sent in f:
                words = sent.split()
                sent_len = len(words)
                # iterate over each inp_word of the sentence
                for i, word in enumerate(words):
                    # get one-hot verctor for the target inp_word
                    if word in self.word2id:
                        trg_w = self.get_one_hot_encoding(word)
                        #logging.info(f"one-hot vec: {trg_w.shape}")

                        # collect indices for the context window
                        ctx_indices = list(range(max(0, i - self.window_size), i)) + list(range(i+1, min(sent_len, i + self.window_size + 1)))
                        ctx_w = []

                        for j in ctx_indices:
                            # add one-hot encodings for all the context words
                            if words[j] in self.word2id:
                                ctx_w.append(self.get_one_hot_encoding(words[j]))

                        if len(ctx_w) > 0:
                            training_data.append([trg_w, ctx_w])

                        if len(training_data) % self.batch_size == 0:
                            training_data = np.asarray(training_data, dtype=object)
                            np.save(self.dir + "batch_training/train_batch" + str(batch) + ".npy", training_data)
                            batch += 1
                            logging.info(f"saved training batch {batch} to {self.dir}batch_training/")
                            training_data = []

        training_data = np.asarray(training_data, dtype=object)
        np.save(self.dir + "batch_training/train_batch" + str(batch) + ".npy", training_data)

    def forward_pass(self, target_wv):
        # target_wv is one-hot vector for target inp_word. inp * wt = hidd
        #logging.info(f"target_wv shape: {target_wv.shape}")
        hidden_layer = np.dot(self.w1.T, target_wv) # shape : (dims,)
        # Dot product hidden layer with second matrix (w2). hidd * wt = u
        output_layer = np.dot(self.w2.T, hidden_layer) # shape: (vocab_size,)
        # predict_prob y' = softmax(u)
        prediction_probabilities = self.softmax(output_layer) # shape (vocab_size,)
        #assert (prediction_probabilities.shape == output_layer.shape)
        return prediction_probabilities, hidden_layer, output_layer

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def train(self):
        # Initialising weight matrices
        self.w1 = np.random.uniform(-1, 1, (self.vocab_size, self.dims))
        self.w2 = np.random.uniform(-1, 1, (self.dims, self.vocab_size))

        epoch_loss = [] # for analysis later

        # for every epoch, go through all the training batch files and all the examples in the batch file
        for i in range(self.n_epochs):
            batch_loss = 0
            b = 0

            for path in Path(self.dir+"batch_training").glob('*.npy'):

                loss = 0  # initialize loss

                #logging.info(f"training batch {b}")
                #str_path = str(path)
                #logging.info(str(b) + path.name)
                b += 1
                training_data = np.load(self.dir+"batch_training/"+path.name, allow_pickle=True)

                # Cycle through each training sample
                # w_t = vector for target inp_word, w_c = vectors for context words
                for w_t, w_c in training_data:
                    # Forward pass - Pass in vector for target inp_word (w_t) to get:
                    #logging.info(f"training_eg {c} in batch {b}")
                    prediction_probab_vector, hidden_layer_matrix, output_layer_matrix = self.forward_pass(w_t)
            
                    # Calculate error
                    # 1. For a target inp_word, calculate difference between y_pred and each of the context words
                    # 2. Sum up the differences using np.sum to get the error for this particular target inp_word
                    error = np.sum([np.subtract(prediction_probab_vector, ctx_wv) for ctx_wv in w_c], axis=0)
                    #logging.info(f"error shape: {error.shape}")
                    # Backpropagation - use SGD to backpropagate errors - calculate loss on the output layer
                    self.backpropogate_error(error, hidden_layer_matrix, w_t)

                    # Calculate loss
                    # Part 1: -ve of the sum of the values in output_layer_matrix (u) that correspond to ctx words +
                    # Part 2: length of context words * log of sum for all elements (exponentiated as per formula) in the output layer (u) before softmax
                    # np.where == 1 returns the index in the context inp_word vector with value 1
                    # Note: output_layer_matrix[np.where(word == 1)] returns the value of the output layer before softmax
                    temp_loss = -np.sum([output_layer_matrix[np.where(word == 1)] for word in w_c]) + \
                                len(w_c) * np.log(np.sum(np.exp(output_layer_matrix)))
                    loss += temp_loss
                logging.info(f"Epoch: {i} Loss: {loss}")
                batch_loss = loss
            epoch_loss.append(batch_loss)

        np.save(self.dir+"w1.npy", self.w1)
        write_obj2file(epoch_loss, self.dir+'epoch_loss.pkl')

    def backpropogate_error(self, error, hidden_layer_matrix, trg_wv):
        # http://www.claudiobellei.com/2018/01/06/backprop-word2vec/#skipgram
        # Going backwards, we need to take derivative of loss with respect of w2
        try:
            dL_dw2 = np.outer(hidden_layer_matrix, error)
            dL_dw1 = np.outer(trg_wv, np.dot(self.w2, error.T))
            #logging.info(f"shape of dL_dw1: {dL_dw1.shape}")
            #logging.info(f"shape w1 {self.w1.shape}")
            # Update weights
            self.w1 = self.w1 - (self.lr * dL_dw1)
            self.w2 = self.w2 - (self.lr * dL_dw2)
        except ValueError as e:
            logging.warning(f"ValueError {e}.\nSkipping this pair for training.")


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help='use this option to provide the file for which embeddings would be generated')
    parser.add_argument("lang",
                        help="specify the lang. abbreviation. of the given file")
    parser.add_argument("--sample_size", default=1000, type=int,
                        help="number of sentences to be considered for training.")
    parser.add_argument("--ctx_window", default=2, type=int,
                        help="size of the context window.")
    parser.add_argument("--dims", default=50, type=int,
                        help="embedding dimension size.")
    parser.add_argument("--generate_training_data", default=True, type=bool,
                        help="set this arg. to False, if training batches have already been created for the given "
                             "sample size and ctx window. Saves computation time!!")
    parser.add_argument("--n_epochs", default=50, type=int,
                        help="number of training epochs")
    parser.add_argument("--lr", default=0.01, type=float,
                        help="training hyperparameter: learning rate")
    parser.add_argument("--batch_size", default=1000, type=int,
                        help="embedding dimension size.")
    parser.add_argument("--temp_dir", default="data/temp_files/",
                        help='directory path to save intermediate files')
    parser.add_argument("--min_freq", default=0, type=int,
                        help="Minimum number of times a word should appear in the corpus to be considered for training.")
    parser.add_argument("--show_logs", default=False, type=bool,
                        help="if False, logs will be saved. if True, they would be shown on console.")

    args = parser.parse_args()
    return args



if __name__ == '__main__':
    # src_file = './data/parallel_corpus/processed_en.src'
    # trg_file = './data/parallel_corpus/processed_de.trg'
    args = arg_parser()
    if not args.show_logs:
        logging.basicConfig(filename="word2vec_training_"+args.lang+".log", filemode='a', format='%(asctime)s - %(message)s',
                        level=logging.INFO)
    else:
        logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

    skipgram = word2vec(args.file, lang=args.lang, sample_size=args.sample_size, window_size=args.ctx_window,
                           dims=args.dims, generate_training_data=args.generate_training_data,
                           n_epochs=args.n_epochs,lr=args.lr, dir=args.temp_dir, batch_size=args.batch_size, min_word_freq=args.min_freq)
    skipgram.train()
