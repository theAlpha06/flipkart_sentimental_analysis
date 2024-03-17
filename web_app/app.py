from flask import Flask, render_template, request
from joblib import load
import nltk
nltk.download('stopwords')
import nltk
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin
from gensim.models import Word2Vec
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from sentence_transformers import SentenceTransformer

import re
import numpy as np

app = Flask(__name__)

class BertVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([self.model.encode(doc) for doc in X])

class GloVeVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, glove_file='glove.6B.100d.txt'):
        self.glove_file = glove_file
        self.word_vectors = self.load_glove_model()

    def load_glove_model(self):
        word2vec_output_file = self.glove_file + '.word2vec'
        glove2word2vec(self.glove_file, word2vec_output_file)
        return KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([self.document_vector(doc) for doc in X])

    def document_vector(self, doc):
        words = self.preprocess_text(doc)
        return np.mean([self.word_vectors[w] for w in words if w in self.word_vectors] or [np.zeros(self.word_vectors.vector_size)], axis=0)


class Word2VecVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, size=100, window=5, min_count=1, workers=4):
        self.size = size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.model = None

    def fit(self, X, y=None):
        tokenised_sentences = [sent.split() for sent in X]
        self.model = Word2Vec(tokenised_sentences, vector_size=self.size, window=self.window, min_count=self.min_count, workers=self.workers)
        return self

    def transform(self, X):
        tokenised_sentences = [sent.split() for sent in X]
        return np.array([self.document_vector(words) for words in tokenised_sentences])

    def document_vector(self, words):
        vocab_tokens = [word for word in words if word in self.model.wv.index_to_key]
        if len(vocab_tokens) == 0:
            return np.zeros(self.model.vector_size)
        else:
            return np.mean(self.model.wv.__getitem__(vocab_tokens), axis=0)

models = {
    "BoW": load('../best_models/naive_bayes.pkl'),
    "TF-IDF": load('../best_models/naive_bayes_tfidf.pkl'),
    "W2V": load('../best_models/logistic_regression_w2v.pkl'),
    "BERT": load('../best_models/logistic_regression_bert.pkl'),
    "Glove": load('../best_models/naive_bayes_glove.pkl'),
}

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'\^[a-zA-Z]\s+', ' ', text) 
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    text = re.sub(r'^b\s+', '', text)
    text = text.lower()
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

def lemmatize_text(text):
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    text = ' '.join(text)
    return text

def prediction_to_label(prediction):
    if prediction == 0:
        return "Negative Review"
    else:
        return "Positive Review"
    
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/result', methods=['GET', 'POST'])
def result():
    if request.method == 'POST':
        review = request.form['review']
        model_name = request.form['model']
        
        review = clean_text(review)
        review = lemmatize_text(review)
        
        model = models[model_name]
        
        prediction = model.predict([review])  
        
        result = prediction_to_label(prediction)
        
        return render_template('index.html', result=result)


if __name__ == "__main__":
    app.run(debug=True)
