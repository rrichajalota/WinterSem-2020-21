import numpy as np
from scipy.sparse import csc_matrix
from collections import Counter
import logging
import re
from pathlib import Path


logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

np.random.seed(42)

class word2vec(object): # skipgram model
    def __init__(self, corpus, sample_size, window_size, dims=100, n_epochs=5, lr= 0.01, dir="data/temp_files/",
                 generate_training_data=True):
        """

        Parameters
        ----------
        corpus - filepath
        dir - path to dir where all temp files will be stored
        sample_size - #samples to be taken from the corpus
        window_size - #context words to consider. choose a value between 2-5
        """
        self.dir = dir
        self.n_epochs = n_epochs
        self.lr = lr
        self.dims = dims
        self.window_size = window_size
        # clean the corpus and save its tokenized version to the temp folder
        self.tokenize(corpus, sample_size)
        # build a vocabulary from the corpus
        self.word_vocab = self.fetch_word_vocab()
        self.vocab_size = len(self.word_vocab)
        self.word2id = self.map_word2id()
        if generate_training_data:
            logging.info(f"Starting training data generation..")
            self.generate_training_data()


    def tokenize(self, corpus, sample_size):
        """
        removes punctuations and digits from the corpus. saves the tokenized text to another file.

        """
        with open(corpus) as f:
            out_file = self.dir+"tokenized_corpus.txt"
            with open(out_file, "w") as tf:
                for i, line in enumerate(f):
                    if i == sample_size: break
                    else:
                        new_line = re.sub('[^A-Za-z ]+', '', line)
                        tf.write(new_line.strip()+"\n")
        logging.info(f"tokenized corpus saved to {out_file}")

    def map_word2id(self):
        word2id = dict()
        for i, word in enumerate(self.word_vocab):
            word2id[word] = i
        logging.info(f"Mapped unique words to ids")
        return word2id

    def fetch_word_vocab(self):
        word_freq = Counter()
        with open(self.dir + "tokenized_corpus.txt") as f:
            for line in f:
                for word in line.split():
                    word_freq[word] += 1
        logging.info(f"number of unique words: {len(word_freq)}")
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

        with open(self.dir + "tokenized_corpus.txt") as f:

            for sent in f:
                words = sent.split()
                sent_len = len(words)
                # iterate over each inp_word of the sentence
                for i, word in enumerate(words):
                    # get one-hot verctor for the target inp_word
                    trg_w = self.get_one_hot_encoding(word)

                    # collect indices for the context window
                    ctx_indices = list(range(max(0, i - self.window_size), i)) + list(range(i+1, min(sent_len, i + self.window_size + 1)))
                    ctx_w = []

                    for j in ctx_indices:
                        # add one-hot encodings for all the context words
                        ctx_w.append(self.get_one_hot_encoding(words[j]))

                    training_data.append([trg_w, ctx_w])

                    if len(training_data) % 500 == 0:
                        training_data = np.asarray(training_data, dtype=object)
                        np.save(self.dir + "batch_training/train_batch" + str(batch) + ".npy", training_data)
                        batch += 1
                        logging.info(f"saved training batch {batch} to {self.dir}/batch_training/")
                        training_data = []

        training_data = np.asarray(training_data, dtype=object)
        np.save(self.dir + "batch_training/train_batch" + str(batch) + ".npy", training_data)

    def forward_pass(self, target_wv):
        # target_wv is one-hot vector for target inp_word
        hidden_layer = np.dot(self.w1.T, target_wv)
        # Dot product hidden layer with second matrix (w2)
        output_layer = np.dot(self.w2.T, hidden_layer) # shape: (vocab_size, )
        # Run 1x9 through softmax to force each element to range of [0, 1] - 1x8
        prediction_probabilities = self.softmax(output_layer)
        assert (prediction_probabilities.shape == output_layer.shape)
        return prediction_probabilities, hidden_layer, output_layer

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def train(self):
        # Initialising weight matrices
        self.w1 = np.random.uniform(-1, 1, (self.vocab_size, self.dims))
        self.w2 = np.random.uniform(-1, 1, (self.dims, self.vocab_size))
        logging.info(f"w1 shape: {self.w1.shape}\t w2 shape: {self.w2.shape}")

        pathlist = Path("data/temp_files/batch_training").glob('*.npy')

        # for every epoch, go through all the training batch files and all the examples in the batch file
        for i in range(self.n_epochs):

            self.loss = 0 # initialize loss

            for path in pathlist:
                str_path = str(path)
                training_data = np.load(str_path, allow_pickle=True)

                # Cycle through each training sample
                # w_t = vector for target inp_word, w_c = vectors for context words
                for w_t, w_c in training_data:
                    # Forward pass - Pass in vector for target inp_word (w_t) to get:
                    # 1. predicted y using softmax (y_pred) 2. matrix of hidden layer (hidden_layer_matrix) 3. output layer before softmax (u)
                    prediction_probab_vector, hidden_layer_matrix, output_layer_matrix = self.forward_pass(w_t)

                    # Calculate error
                    # 1. For a target inp_word, calculate difference between y_pred and each of the context words
                    # 2. Sum up the differences using np.sum to give us the error for this particular target inp_word
                    error = np.sum([np.subtract(prediction_probab_vector, ctx_wv) for ctx_wv in w_c], axis=0)

                    # Backpropagation - use SGD to backpropagate errors - calculate loss on the output layer
                    self.backprop(error, hidden_layer_matrix, w_t)

                    # Calculate loss
                    # There are 2 parts to the loss function
                    # Part 1: -ve sum of all the output +
                    # Part 2: length of context words * log of sum for all elements (exponential-ed) in the output layer before softmax (u)
                    # Note: inp_word.index(1) returns the index in the context inp_word vector with value 1
                    # Note: output_layer_matrix[inp_word.index(1)] returns the value of the output layer before softmax
                    self.loss += -np.sum([output_layer_matrix[np.where(word == 1)] for word in w_c]) + len(w_c) * np.log(np.sum(np.exp(output_layer_matrix)))
            logging.info(f"Epoch: {i} Loss: {self.loss}")

        np.save("data/temp_files/w1.npy", self.w1)
        np.save("data/temp_files/w2.npy", self.w2)

    def backprop(self, error, hidden_layer_matrix, trg_wv):
        # Column vector error represents row-wise sum of prediction errors across each context inp_word for the current center inp_word
        # Going backwards, we need to take derivative of E with respect of w2
        dl_dw2 = np.outer(hidden_layer_matrix, error)
        # target_wv - shape 1x8, w2 - 5x8, error.T - 8x1
        # target_wv - 1x8, np.dot() - 5x1, dl_dw1 - 8x5
        dl_dw1 = np.outer(trg_wv, np.dot(self.w2, error.T))
        # Update weights
        self.w1 = self.w1 - (self.lr * dl_dw1)
        self.w2 = self.w2 - (self.lr * dl_dw2)

    # Get vector from inp_word
    def word_vec(self, word):
        w_index = self.word2id[word]
        v_w = self.w1[w_index]
        return v_w

    # Input vector, returns nearest inp_word(s)
    def vec_sim(self, inp_word, top_n):
        v_w1 = self.word_vec(inp_word)
        word_sim = {}

        for i in range(self.vocab_size):
            # Find the similary score for each inp_word in vocab
            v_w2 = self.w1[i]
            theta_sum = np.dot(v_w1, v_w2)
            theta_den = np.linalg.norm(v_w1) * np.linalg.norm(v_w2)
            theta = theta_sum / theta_den

            word = self.word_vocab[i]
            word_sim[word] = theta

        words_sorted = sorted(word_sim.items(), key=lambda kv: kv[1], reverse=True)

        for wd, sim in words_sorted[:top_n+1]:
            if inp_word != wd:
                print(wd, sim)

def cross_entropy(softmax_out, Y):
    """
    softmax_out: output out of softmax. shape: (vocab_size, m)
    """
    m = softmax_out.shape[1]
    cost = -(1 / m) * np.sum(np.sum(Y * np.log(softmax_out + 0.001), axis=0, keepdims=True), axis=1)
    return cost



if __name__ == '__main__':
    src_file = './data/parallel_corpus/processed_en.src'
    skipgram = word2vec(src_file, sample_size=10000, window_size=3, dims=50, generate_training_data=False)
    skipgram.train()
    skipgram.vec_sim("i",top_n=5)