import re

import nltk
import pandas as pd
from nltk.corpus import stopwords

class Data_Preprocessor:
    def __init__(self):
        self.stopwords = set(stopwords.words('english'))

    def remove_stopwords_shortwords(self, tokens):
        res = []
        for token in tokens:
            if not token in self.stopwords and len(token) > 2:
                res.append(token)
        return res

    def remove_special_symbols(self, text):
        '''
        removes arabic, tamil, latin symbols and dingbats
        :param text:
        :return:
        '''
        special_symbols = re.compile(r"[\u0600-\u06FF\u0B80-\u0BFF\u25A0-\u25FF\u2700-\u27BF]+", re.UNICODE)
        text = special_symbols.sub('', text)
        other_symbols = re.compile(r'([#&@.]+)')
        text = other_symbols.sub('', text)
        return text


    def process_text(self,text):
        text = text.encode('ascii', errors='ignore').decode()
        text = text.lower()
        text = re.sub(r'http\S+', ' ', text)
        text = re.sub(r'#+', ' ', text)
        text = re.sub(r'@[A-Za-z0-9]+', ' ', text)
        text = re.sub(r"([A-Za-z]+)'s", r"\1 is", text)
        #text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"won't", "will not ", text)
        text = re.sub(r"isn't", "is not ", text)
        text = re.sub(r"can't", "can not ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub('\W', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        text = re.sub('\s+', ' ', text)
        text = re.sub('rt', ' ', text)
        text = self.remove_special_symbols(text)
        text = text.strip()
        return text

    def process_all(self,text):
        text = self.process_text(text)
        return ' '.join(self.remove_stopwords_shortwords(text.split()))


def main():
    preprocesser = Data_Preprocessor()
    filepath = 'australiaBushfire2013_tweets.txt' # downloaded from the trec-is challenge site
    data = pd.read_csv(filepath,sep=',',header=0)
    tweets = data['full_text'].tolist()
    with open('data/pptweets_chileEarthquake2014_australiaBushfire2013.txt', 'w') as f:
        for i, tweet in enumerate(tweets):
            if isinstance(tweet, str):
                tweets[i] = preprocesser.process_all(tweet)
                f.write(tweets[i]+"\n")


if __name__ == '__main__':
    main()
