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
        """
        Parameters
        ----------
        docs - list of wordlist [[..]]
        ntopics - num_topics to find
        word_vocab - list of unique words in the corpus
        alpha - gibbs sampler param
        beta - gibbs sampler param
        """
        self.docs = docs
        self.ntopics = ntopics
        self.word_vocab = word_vocab
        self.alpha = alpha
        self.beta = beta

        # ---- declare derived variables ----
        self.vocab_size = len(word_vocab)
        self.doc_len = len(docs)
        self.word2topic: ndarray = np.zeros((self.vocab_size, ntopics)) #shape (vocab_size, ntopics)
        self.doc2topic_count = np.zeros((self.doc_len, ntopics)) # shape (doc_len, ntopics)
        self.num_topics_in_doc = np.zeros((self.doc_len,)) # shape (doc_len,)
        self.num_topic_occurence = np.zeros((ntopics,)) # shape (ntopics,)
        self.topic_assignment = defaultdict(list) # {doc_index: [topic1, topic2, .., topic1] (list_length = len(doc))}
        self.topiclist = np.arange(ntopics) # shape (ntopics,)

    def initialize_counts(self):
        """
        Allots topics randomly to all words in the corpus. Then increments the count matrices as per topic allotment.
        """
        start = time()

        for m, doc in enumerate(self.docs):
            self.topic_assignment[m] = [-1] * len(doc)
            for n, word in enumerate(doc):
                k = np.random.randint(low=0, high=self.ntopics - 1) # sample topic index

                self.topic_assignment[m][n] = k
                self.doc2topic_count[m, k] += 1
                self.word2topic[word, k] += 1
                self.num_topic_occurence[k] += 1
                self.num_topics_in_doc[m] += 1

        logging.info(f'time taken to initialize counts: {time()-start} secs.')

    def initialize_counts_with_same_topic_per_doc(self):
        """
        Allots same topics to all words in a doc. Then increments the count matrices as per topic allotment.
        """
        start = time()

        for m, doc in enumerate(self.docs):
            k = np.random.randint(low=0, high=self.ntopics - 1)
            num_words_in_doc = len(doc)

            self.topic_assignment[m] = [k] * num_words_in_doc # all words get the same topic
            self.doc2topic_count[m, k] += num_words_in_doc # a
            self.num_topics_in_doc[m] += num_words_in_doc
            self.num_topic_occurence[k] += num_words_in_doc

        logging.info(f'time taken to initialize counts: {time()-start} secs.')


    def perform_lda(self, niterations, non_random_init):
        """
        Parameters
        ----------
        niterations - number of iterations to perform
        non_random_init - True/False. if True, a non-random topic assignment strategy is set.

        Returns
        -------
        word2topic matrix of shape (vocab_size x ntopics)

        """
        # 1) perform initialization
        if non_random_init is True:
            self.initialize_counts_with_same_topic_per_doc()
        else:
            self.initialize_counts()

        # 2) for every word in every doc, for k iterations
        for _ in tqdm(range(0, niterations)):

            for m, doc in enumerate(self.docs):
                self.num_topics_in_doc[m] -= 1 # In every inner for-loop interation, we exclude the current word from the counts,
                                               # hence this optimization step: instead of incrementing and decrementing inside the inner for-loop
                                               # update this vector here.
                for n, word in enumerate(doc):
                    topic = self.topic_assignment[m][n]  # get the current topic assignment of the word in the doc.
                    self.decrement_counts_for_topic(m, topic, word)  # for the current assignment of k to a term t for word Wm,n:
                                                                     # decrement counts and sums
                    condprob_z = self.compute_probdist_over_topics(m, word)    # compute conditional probability

                    topic = np.random.choice(self.topiclist, p=condprob_z) # sample from the newly computed probability dist.

                    self.topic_assignment[m][n] = topic # assign new topic to the word
                    self.increment_counts_for_topic(m, topic, word) # increment counts for the new topic

                self.num_topics_in_doc[m] += 1

        return self.word2topic

    def increment_counts_for_topic(self, m, topic, word):
        self.doc2topic_count[m,topic] += 1
        self.word2topic[word,topic] += 1
        self.num_topic_occurence[topic] += 1


    def compute_probdist_over_topics(self, doc_id, word):
        """

        Parameters
        ----------
        doc_id - index of the doc
        word - int value that refers to the index of the word in the word_vocab (list of uniq words)

        Returns
        -------
        np.ndarray of shape (ntopics,)
        """
        condprob_z = ((self.doc2topic_count[doc_id, :] + self.alpha) / (self.num_topics_in_doc[doc_id] + (self.ntopics * self.alpha))) * \
                     ((self.word2topic[word, :] + self.beta) / (self.num_topic_occurence + (self.beta * self.vocab_size)))
        condprob_z /= np.sum(condprob_z) # (ntopics,) / scalar = (ntopics,)
        return condprob_z

    def decrement_counts_for_topic(self, doc_id, topic, word):
        """
        max(0, count-1) operation helps with the alternate topic assignment strategy, prevents the values in
        doc2topic/word2topic from becoming negative.
        However, it leads to an overhead of ~1.5 secs/iteration in runtime on 2000 samples of movie-reviews dataset.
        Parameters
        ----------
        doc_id - index of doc
        topic - int value that refers to the topic assigned
        word - int value that refers to the index of the word in the word_vocab (list of uniq words)

        Returns
        -------

        """
        self.doc2topic_count[doc_id, topic] = max(0, self.doc2topic_count[doc_id, topic] - 1)
        self.word2topic[word, topic] = max(0, self.word2topic[word, topic]-1)
        self.num_topic_occurence[topic] -= 1
        # self.doc2topic_count[doc_id,topic] -= 1
        # self.word2topic[word,topic] -= 1
        # self.num_topic_occurence[topic] -= 1


def main():
    args = arg_parser()

    filepath = args.filepath if args.filepath else './moviespp.txt/movies-pp.txt'
    num_samples = args.num_samples if args.num_samples else None

    # print out the default/set arguments.
    show_arguments(args, filepath=filepath, num_samples=num_samples)

    with open(filepath) as f:
        data = f.readlines() # read all 2000 lines

        if not args.include_first_line: # assuming line 1 contains header info
            data = data[1:]

        if num_samples is not None: # allows to take only a subset of samples from the dataset
            if num_samples > len(data):
                raise ValueError(f'The specified value for num_samples {num_samples} should be less than or equal '
                                 f'to the dataset length {len(data)}.')
            data = data[:num_samples]

        docs, word_vocab = fetch_docVec_wordVocab(data) # docs = [[ 'he', 'is'..],['why', 'is'..],]
        logging.info(f'num_samples being used: {len(docs)}\t num_unique_words: {len(word_vocab)}')

        word2id = assign_wordIDs(word_vocab)
        docs_with_word_ids = [[word2id[word] for word in wordlist] for wordlist in docs]  # replace words in docs with their word IDs
                            # docs_with_word_ids = [[3, 1,..], [2, 1,..],]

        model = TopicModel(docs=docs_with_word_ids, ntopics=args.ntopics, word_vocab=word_vocab, alpha=args.alpha, beta=args.beta)

        word2topic = model.perform_lda(niterations=args.niterations, non_random_init=args.non_random_init)


        print_output(args, word2topic, word_vocab) # print out words from each topic


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
    parser.add_argument("--non_random_init", default=False, action='store_true',
                        help='Specify this argument for a non-random word initialization of topics. '
                             'When specified, same topic is given to all words of the same document.')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()

