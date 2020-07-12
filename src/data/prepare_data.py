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
from data.data_utils import clean_strings

df = pd.read_csv('../data/raw/tripadvisor_reviews.csv')

# Convertir a una lista
data = df.review.values.tolist()

# Eliminar emails
data = clean_strings(data)

# Save in plk files
with open(r"../data/interim/prepared_data.pkl", "wb") as output_file:
    pickle.dump(data, output_file)