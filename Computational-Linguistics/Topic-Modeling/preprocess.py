import re
import nltk

class Data_Preprocessor:

    def remove_stopwords(self,stop_words, tokens):
        res = []
        for token in tokens:
            if not token in stop_words:
                res.append(token)
        return res


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
        text = text.strip()
        return text

    def process_all(self,text,stop_words):
        text = self.process_text(text)
        return ' '.join(self.remove_stopwords(stop_words, text.split()))

def main():
