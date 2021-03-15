import pandas as pd
from mosestokenizer import MosesTokenizer
import nltk
import re
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


def create_parallel_files(tsv_file, out_dir="data/parallel_corpus/", stemming=False):
    """
    This fn. tokenizes and stems en and de words, lowercases them and writes them to two separate txt files.
    """
    col_names = ['de', 'en']
    en_stemmer = nltk.stem.SnowballStemmer('english')
    de_stemmer = nltk.stem.SnowballStemmer('german')
    de_regex = '[0-9!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~]+'
    en_regex = '[^A-Za-z ]+'

    de_tokenizer = MosesTokenizer(lang='de')
    en_tokenizer = MosesTokenizer(lang='en')
    data = pd.read_csv(tsv_file, sep="\t", header=None, names=col_names)
    print(len(data.de))
    print(len(data.en))
    with open(out_dir+'processed_de.trg', 'w') as tf, open(out_dir+'processed_en.src', 'w') as sf,\
            open(out_dir+'fastAlign_inp_en-de.txt', 'w') as fa:
        for index, row in data.iterrows():
            de_sent = str(row.de).replace("\n", " ")
            en_sent = str(row.en).replace("\n", " ")
            de_sent = de_tokenizer(de_sent.strip())
            en_sent = en_tokenizer(en_sent.strip())
            if stemming is True:
                de_sent = " ".join([de_stemmer.stem(wd) for wd in de_sent]).lower()
                en_sent = " ".join([en_stemmer.stem(wd) for wd in en_sent]).lower()
            else:
                de_sent = " ".join(de_sent).lower()
                en_sent = " ".join(en_sent).lower()
            clean_de_sent = remove_nonword_chars(de_sent, de_regex)
            clean_en_sent = remove_nonword_chars(en_sent, en_regex)
            if len(clean_de_sent) == 0 or len(clean_en_sent) == 0:
                logging.info(f"blank line index: {index}\n en:{clean_en_sent} \n de: {clean_de_sent}")
                continue
            tf.write(clean_de_sent+"\n")
            sf.write(clean_en_sent+"\n")
            fa.write(clean_en_sent + " ||| " + clean_de_sent+"\n")

def remove_nonword_chars(line, regex):
    new_line = re.sub(regex, '', line)
    new_line = re.sub(' +', ' ', new_line)  # remove multiple spaces between words, if such spaces exist
    return new_line 
    


if __name__ == '__main__':
    path = 'data/parallel_corpus/trimmed_de_en.txt'
    create_parallel_files(path, stemming=False) # from the tab separated text file
    #src_file = 'data/parallel_corpus/processed_en.src'
    #trg_file = "data/parallel_corpus/processed_de.trg"
    
    #remove_nonword_chars(src_file, en_regex)
    #remove_nonword_chars(trg_file, de_regex)

    