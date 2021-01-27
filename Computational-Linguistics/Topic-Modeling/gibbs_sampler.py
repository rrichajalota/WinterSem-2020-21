from numpy.core._multiarray_umath import ndarray
from tqdm import tqdm
import numpy as np
import argparse
from pprint import pprint
from time import time
from collections import Counter, defaultdict
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


def perform_lda(docs ,niterations, ntopics, num_word_types, alpha, beta):
    """
    Parameters
    ----------
    docs - list of docs containing wordlists (sentences)
    niterations - num_epochs
    ntopics - int
    num_word_types - Counter containing number of times a word_type occurs in the corpus, doc. eg. {'she':5, 'he':2..}
    """
    # initialize count matrices, word2topic of shape (len(num_word_types), ntopics) -- nkt
    # doc2topic of shape (len(docs), ntopics) -- nmk
    # num_topic_occuerence (nk) = [0] * ntopics -> num times any word is assigned to topic k
    # topic_assignment (Zmn) array of shape (len(docs), len(doc[i]) where i = [0, len(docs))
    #start = time()
    vocab_size = len(num_word_types)

    word2topic: ndarray = np.zeros((vocab_size, ntopics), dtype=int)
    doc2topic_count = np.zeros((len(docs), ntopics), dtype=int)
    num_topics_in_doc = np.zeros((len(docs),), dtype=int)
    num_topic_occurence = np.zeros((ntopics,), dtype=int)
    topic_assignment = [[None for _ in sent] for sent in docs]

    # ---- initiaization loop ----- #
    for m, doc in enumerate(docs):
        for n, word in enumerate(doc):
            # sample topic index
            k = np.random.random_integers(low=0, high=ntopics-1)
            topic_assignment[m][n] = k
            doc2topic_count[m][k] += 1
            num_topics_in_doc[m] += 1
            word2topic[word][k] += 1
            num_topic_occurence[k] += 1
    # end2 = time()
    # logging.info(f'time taken for initialization: {end2-end1} secs.')
    # gibbs sampling
    for _ in tqdm(range(0, niterations)):
        #start = time()
        for m, doc in enumerate(docs):
            num_topics_in_doc[m] -= 1

            for n, word in enumerate(doc):
                # for the current assignment of k to a term t for word Wm,n:
                # decrement counts and sums
                topic = topic_assignment[m][n]
                doc2topic_count[m][topic] -= 1
                word2topic[word][topic] -= 1
                num_topic_occurence[topic] -= 1

                # compute conditional probability
                condprob_z = ((doc2topic_count[m, :] + alpha) * (word2topic[word, :] + beta)) /  ((num_topic_occurence + (beta * vocab_size)) * (num_topics_in_doc[m] + ntopics*alpha))
                condprob_z /= np.sum(condprob_z)
                # p2 = time()
                # logging.info(f'time taken to calc. condprob for one word: {p2-p1} secs')
                topic = np.random.choice(np.arange(ntopics), p=condprob_z)
                # p3 = time()
                # logging.info(f'time taken to calc. new_topic for one word: {p3-p2} secs')
                topic_assignment[m][n] = topic
                doc2topic_count[m][topic] += 1
                word2topic[word][topic] += 1
                num_topic_occurence[topic] += 1

            num_topics_in_doc[m] += 1
        #logging.info(f'one iteration takes: {time()-start} secs.')
    return topic_assignment, doc2topic_count, word2topic, num_topic_occurence


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", help='use this option to provide a file/corpus for topic modeling.')
    parser.add_argument("--num_samples",
                        help="use this option to state the number of samples to take from the given file")
    args = parser.parse_args()
    return args


def assign_wordIDs(uniq_wordlist):
    word2id = dict()
    for i, word in enumerate(uniq_wordlist):
        word2id[word] = i
    return word2id


def main():
    args = arg_parser()

    filepath = args.filepath if args.filepath else './moviespp.txt/movies-pp.txt'
    num_samples = args.num_samples if args.num_samples else 2000

    with open(filepath) as f:
        data = f.readlines() # read all 2000 lines
        # data = ['play football holiday',
        #         'I like watch football holiday',
        #         'la liga match holiday',
        #         'vehicle rightly advertised smooth silent',
        #         'vehicle good pickup very comfortable drive',
        #         'mileage vehicle around 14kmpl',
        #         'vehicle 7 seater MPV generates 300 Nm torque with diesel engine']
        docs = []
        num_word_types = Counter()
        for i, sentence in tqdm(enumerate(data)):
            if i == 0: continue
            #if i == num_samples+1: break
            words = sentence.split()
            docs.append(words)
            for word in words:
                num_word_types[word] +=1

        word_vocab = list(num_word_types.keys())
        word2id = assign_wordIDs(word_vocab)

        logging.info(f'num_samples: {num_samples} \n num_uniq_words: {len(word_vocab)}')

        # replace words from docs with word IDs
        docs = [[word2id[word] for word in wordlist] for wordlist in docs]
        num_word_types = {word2id[k]:v for k, v in num_word_types.items()}

        topic_assignment, doc2topic_count, word2topic, num_topic_occurence = perform_lda(docs, niterations=500, ntopics=20, num_word_types=num_word_types, alpha=0.02, beta=0.1)

        # print out words from each topic
        topic2word = word2topic.T
        for t, topic in enumerate(topic2word):
            words_in_topic = Counter()
            for w, wc in enumerate(topic):
                if wc!=0:
                    words_in_topic[word_vocab[w]] = wc
            print(f'\n#---------------------------------------#\n Topic {t} \n {words_in_topic.most_common(25)}')










if __name__ == '__main__':
    main()

