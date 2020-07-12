import re
import numpy as np
import pandas as pd
from pprint import pprint
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from nltk.corpus import stopwords
import spacy
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
import pickle

import nltk
from features.utils import sent_to_words
from features.utils import remove_stopwords
from features.utils import make_bigrams
from features.utils import make_trigrams
from features.utils import lemmatization
from features.utils import get_corpus_info

# Load data
data = None
with open(r"../data/interim/prepared_data.pkl", "rb") as input_file:
    data = pickle.load(input_file)

corpus_info = get_corpus_info(data, load_bigram = False)

# Save in plk files
with open(r"../data/interim/corpus.pkl", "wb") as output_file:
    pickle.dump(corpus_info['corpus'], output_file)

with open(r"../data/interim/id2word.pkl", "wb") as output_file:
    pickle.dump(corpus_info['id2word'], output_file)

with open(r"../data/interim/data_lemmatized.pkl", "wb") as output_file:
    pickle.dump(corpus_info['data_lemmatized'], output_file)