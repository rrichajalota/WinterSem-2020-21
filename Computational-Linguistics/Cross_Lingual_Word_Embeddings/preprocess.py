import pandas as pd
from mosestokenizer import MosesTokenizer
import nltk

def create_parallel_files(tsv_file):
    col_names = ['de', 'en']
    en_stemmer = nltk.stem.SnowballStemmer('english')
    de_stemmer = nltk.stem.SnowballStemmer('german')

    de_tokenizer = MosesTokenizer(lang='de')
    en_tokenizer = MosesTokenizer(lang='en')
    data = pd.read_csv(tsv_file, sep="\t", header=None, names=col_names)
    print(len(data.de))
    print(len(data.en))
    with open('data/parallel_corpus/processed_de.trg', 'w') as tf, open('data/parallel_corpus/processed_en.src', 'w') as sf:
        for index, row in data.iterrows():
            de_sent = str(row.de).replace("\n", " ")
            en_sent = str(row.en).replace("\n", " ")
            de_sent = de_tokenizer(de_sent.strip())
            en_sent = en_tokenizer(en_sent.strip())
            de_sent = " ".join([de_stemmer.stem(wd) for wd in de_sent]).lower()
            en_sent = " ".join([en_stemmer.stem(wd) for wd in en_sent]).lower()
            tf.write(de_sent+"\n")
            sf.write(en_sent+"\n")


if __name__ == '__main__':
    path = 'data/tmxt/trimmed_de_en.txt'
    create_parallel_files(path)
    