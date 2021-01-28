from numpy.core._multiarray_umath import ndarray
from tqdm import tqdm
import numpy as np
import argparse
from pprint import pprint
from time import time
from collections import Counter, defaultdict
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

np.random.seed(123)

class TopicModel(object):
    def __init__(self, docs, ntopics, word_vocab, alpha, beta):
        self.docs = docs
        self.ntopics = ntopics
        self.word_vocab = word_vocab
        self.alpha = alpha
        self.beta = beta

        # ---- declare derived variables ----
        self.vocab_size = len(word_vocab)
        self.doc_len = len(docs)
        self.word2topic: ndarray = np.zeros((self.vocab_size, ntopics), dtype=np.int32) #shape (vocab_size, ntopics)
        self.doc2topic_count = np.zeros((self.doc_len, ntopics), dtype=np.int32) # shape (doc_len, ntopics)
        self.num_topics_in_doc = np.zeros((self.doc_len,), dtype=np.int32) # shape (doc_len,)
        self.num_topic_occurence = np.zeros((ntopics,), dtype=np.int32) # shape (ntopics,)
        self.topic_assignment = [[None for _ in sent] for sent in docs] # variable-column-sized 2D python vector: doc_len rows
        self.topiclist = np.arange(ntopics) # shape (ntopics,)

    def initialize_counts(self):
        for m, doc in enumerate(self.docs):
            self.num_topics_in_doc[m] += len(doc)
            for n, word in enumerate(doc):
                k = np.random.randint(low=0, high=self.ntopics - 1) # sample topic index
                self.topic_assignment[m][n] = k
                self.doc2topic_count[m][k] += 1
                self.word2topic[word][k] += 1
                self.num_topic_occurence[k] += 1

    def initialize_counts_with_same_topic_per_doc(self):
        for m, doc in enumerate(self.docs):
            k = np.random.randint(low=0, high=self.ntopics - 1)
            num_words_in_doc = len(doc)
            # assign the same topic to all words of the same document
            self.topic_assignment[m] = [k] * num_words_in_doc
            self.doc2topic_count[m][k] += num_words_in_doc
            self.num_topics_in_doc[m] += num_words_in_doc
            self.num_topic_occurence[k] += num_words_in_doc

    def perform_lda(self, niterations):
        self.initialize_counts()
        #self.initialize_counts_with_same_topic_per_doc()

        for _ in tqdm(range(0, niterations)):

            for m, doc in enumerate(self.docs):
                self.num_topics_in_doc[m] -= 1

                for n, word in enumerate(doc):
                    topic = self.topic_assignment[m][n]
                    self.decrement_counts_for_topic(m, topic, word)  # for the current assignment of k to a term t for word Wm,n:
                                                                     # decrement counts and sums
                    condprob_z = self.compute_probdist_over_topics(m, word)    # compute conditional probability
                    #condprob_z = self.compute_log_probdist_over_topics(m, word)

                    topic = np.random.choice(self.topiclist, p=condprob_z) # sample from the newly computed probability dist.

                    self.topic_assignment[m][n] = topic # assign new topic to the word
                    self.increment_counts_for_topic(m, topic, word) # increment counts for the new topic

                self.num_topics_in_doc[m] += 1

        return self.topic_assignment, self.doc2topic_count, self.word2topic, self.num_topic_occurence

    def increment_counts_for_topic(self, m, topic, word):
        self.doc2topic_count[m,topic] += 1
        self.word2topic[word,topic] += 1
        self.num_topic_occurence[topic] += 1


    def compute_probdist_over_topics(self, m, word):
        condprob_z = ((self.doc2topic_count[m, :] + self.alpha)  / (self.num_topics_in_doc[m] + (self.ntopics * self.alpha))) *\
                     ((self.word2topic[word, :] + self.beta) / (self.num_topic_occurence + (self.beta * self.vocab_size)))
        condprob_z /= np.sum(condprob_z)
        return condprob_z

    def decrement_counts_for_topic(self, m, topic, word):
        # self.doc2topic_count[m, topic] = max(0, self.doc2topic_count[m, topic]-1)
        # self.word2topic[word, topic] = max(0, self.word2topic[word, topic]-1)
        # self.num_topic_occurence[topic] -= 1
        self.doc2topic_count[m,topic] -= 1
        self.word2topic[word,topic] -= 1
        self.num_topic_occurence[topic] -= 1


# def perform_lda(docs, niterations, ntopics, word_vocab, alpha, beta):
#     """
#     Parameters
#     ----------
#     docs - list of docs containing wordlists (sentences)
#     niterations - num_epochs
#     ntopics - int
#     word_vocab - list containing unique words in the vocab
#     """
#     vocab_size = len(word_vocab)
#
#     word2topic: ndarray = np.zeros((vocab_size, ntopics), dtype=np.int32)
#     doc2topic_count = np.zeros((len(docs), ntopics), dtype=np.int32)
#     num_topics_in_doc = np.zeros((len(docs),), dtype=np.int32)
#     num_topic_occurence = np.zeros((ntopics,), dtype=np.int32)
#     topic_assignment = [[None for _ in sent] for sent in docs]
#     topiclist = np.arange(ntopics)
#
#     # ---- initiaization loop ----- #
#     for m, doc in enumerate(docs):
#         for n, word in enumerate(doc):
#             # sample topic index
#             k = np.random.random_integers(low=0, high=ntopics-1)
#             topic_assignment[m][n] = k
#             doc2topic_count[m][k] += 1
#             num_topics_in_doc[m] += 1
#             word2topic[word][k] += 1
#             num_topic_occurence[k] += 1
#
#
#     for _ in tqdm(range(0, niterations)):
#         #start = time()
#         for m, doc in enumerate(docs):
#             num_topics_in_doc[m] -= 1
#
#             for n, word in enumerate(doc):
#                 # for the current assignment of k to a term t for word Wm,n:
#                 # decrement counts and sums
#                 topic = topic_assignment[m][n]
#                 doc2topic_count[m][topic] -= 1
#                 word2topic[word][topic] -= 1
#                 num_topic_occurence[topic] -= 1
#
#                 # compute conditional probability
#                 condprob_z = ((doc2topic_count[m, :] + alpha) * (word2topic[word, :] + beta)) /  \
#                              ((num_topic_occurence + (beta * vocab_size)) * (num_topics_in_doc[m] + ntopics*alpha))
#                 condprob_z /= np.sum(condprob_z)
#                 # p2 = time()
#                 # logging.info(f'time taken to calc. condprob for one word: {p2-p1} secs')
#                 topic = np.random.choice(topiclist, p=condprob_z)
#                 # p3 = time()
#                 # logging.info(f'time taken to calc. new_topic for one word: {p3-p2} secs')
#                 topic_assignment[m][n] = topic
#                 doc2topic_count[m][topic] += 1
#                 word2topic[word][topic] += 1
#                 num_topic_occurence[topic] += 1
#
#             num_topics_in_doc[m] += 1
#         #logging.info(f'one iteration takes: {time()-start} secs.')
#     return topic_assignment, doc2topic_count, word2topic, num_topic_occurence


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", help='use this option to provide a file/corpus for topic modeling.'
                                           'By default, samples from second line onwards are considered '
                                           '(assuming line 1 gives header info). To change this behaviour, '
                                           'use --include_first_line.')
    parser.add_argument("--num_samples", type=int,
                        help="use this option to state the number of samples to take from the given file")
    parser.add_argument("--include_first_line", default=False, action='store_true', dest='include_first_line',
                        help="if specified, includes first line of the dataset.")
    parser.add_argument("--alpha", default=0.02, type=float,
                        help="specify the hyperparameter for Dirichlet distribution. (default 0.02)")
    parser.add_argument("--beta", default=0.1, type=float,
                        help="specify the hyperparameter for Dirichlet distribution. (default 0.1)")
    parser.add_argument("--niterations", type=int,
                        default=500, help="num of iterations to run (default 500)")
    parser.add_argument("--ntopics", type=int,
                        default=20, help="num of topics to find (default 20)")
    parser.add_argument("--show_words", type=int,
                        default=40, help="number of most frequent words to show (default 40)")
    args = parser.parse_args()
    return args


def assign_wordIDs(uniq_wordlist):
    word2id = dict()
    for i, word in enumerate(uniq_wordlist):
        word2id[word] = i
    return word2id


def show_arguments(args, num_samples, filepath):
    logging.info(f'filepath: {filepath}\nspecified num_samples: {num_samples}\n'
                 f'include_first_line: {args.include_first_line}\n'
                 f'alpha: {args.alpha}\nbeta: {args.beta}\n'
                 f'number of interations: {args.niterations}\n'
                 f'num topics to find: {args.ntopics} \n'
                 f'num words to show per topic: {args.show_words}')


def main():
    args = arg_parser()

    filepath = args.filepath if args.filepath else './moviespp.txt/movies-pp.txt'
    num_samples = args.num_samples if args.num_samples else None

    show_arguments(args, filepath=filepath, num_samples=num_samples)

    with open(filepath) as f:
        data = f.readlines() # read all 2000 lines

        if not args.include_first_line:
            data = data[1:]

        if num_samples is not None:
            if num_samples > len(data):
                raise ValueError(f'The specified value for num_samples {num_samples} should be less than or equal '
                                 f'to the dataset length {len(data)}.')
            data = data[:num_samples]

        docs, word_vocab = fetch_docVec_wordVocab(data)
        logging.info(f'num_samples being used: {len(docs)}\t num_unique_words: {len(word_vocab)}')

        word2id = assign_wordIDs(word_vocab)
        docs_with_word_ids = [[word2id[word] for word in wordlist] for wordlist in docs]  # replace words in docs with their word IDs

        model = TopicModel(docs=docs_with_word_ids, ntopics=args.ntopics, word_vocab=word_vocab, alpha=args.alpha, beta=args.beta)

        topic_assignment, doc2topic_count, word2topic, num_topic_occurence = model.perform_lda(niterations=args.niterations)


        # topic_assignment, doc2topic_count, word2topic, num_topic_occurence = perform_lda(docs=docs_with_word_ids,
        #                                                                                  niterations=args.niterations,
        #                                                                                  ntopics=args.ntopics,
        #                                                                                  word_vocab=word_vocab,
        #                                                                                  alpha=args.alpha, beta=args.beta)
        print_output(args, word2topic, word_vocab) # print out words from each topic


def print_output(args, word2topic, word_vocab):
    topic2word = word2topic.T
    for t, topic in enumerate(topic2word):
        words_in_topic = Counter()
        for w, wc in enumerate(topic):
            if wc != 0:
                words_in_topic[word_vocab[w]] = wc
        print(
            f'\n#---------------------------------------#\n Topic {t} \n {words_in_topic.most_common(args.show_words)}')


def fetch_docVec_wordVocab(data):
    docs = []
    word_vocab = set()  # Counter()
    for i, sentence in enumerate(data):
        words = sentence.split()
        docs.append(words)
        for word in words:
            word_vocab.add(word)  # [word] +=1
    word_vocab = list(word_vocab)
    return docs, word_vocab


if __name__ == '__main__':
    main()

