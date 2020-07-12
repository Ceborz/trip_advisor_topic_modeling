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

def sent_to_words(sentences):
    for sentence in sentences:
        # https://radimrehurek.com/gensim/utils.html#gensim.utils.simple_preprocess
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True elimina la puntuación

# Eliminar stopwords
def remove_stopwords(texts, stop_words):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

# Hacer bigrams
def make_bigrams(texts, bigram_mod):
    return [bigram_mod[doc] for doc in texts]

# Hacer trigrams
def make_trigrams(texts, bigram_mod, trigram_mod):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

# Lematización basada en el modelo de POS de Spacy
def lemmatization(texts, nlp, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

def get_corpus_info(data, load_bigram = True):
    nltk.download('stopwords')

    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

    data_words = list(sent_to_words(data))

    if load_bigram:
        with open('models/bigrams.pkl', 'rb') as input_file:
            bigram_mod = pickle.load(input_file)
        with open('models/trigrams.pkl', 'rb') as input_file:
            trigram_mod = pickle.load(input_file)
    else:
        # Construimos modelos de bigrams y trigrams
        # https://radimrehurek.com/gensim/models/phrases.html#gensim.models.phrases.Phrases
        bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
        trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

        # Aplicamos el conjunto de bigrams/trigrams a nuestros documentos
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        trigram_mod = gensim.models.phrases.Phraser(trigram)

        # Save n-grams
        with open(r"models/bigrams.pkl", "wb") as output_file:
            pickle.dump(bigram_mod, output_file)
        with open(r"models/trigrams.pkl", "wb") as output_file:
            pickle.dump(trigram_mod, output_file)

    # Eliminamos stopwords
    data_words_nostops = remove_stopwords(data_words, stop_words)
    # Formamos bigrams
    data_words_bigrams = make_bigrams(data_words_nostops, bigram_mod)
    # python3 -m spacy download en_core_web_lg
    # Inicializamos el modelo 'en_core_web_lg' con las componentes de POS únicamente
    nlp = spacy.load('en_core_web_lg', disable=['parser', 'ner'])

    # Lematizamos preservando únicamente noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_bigrams, nlp, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    # Creamos diccionario
    id2word = corpora.Dictionary(data_lemmatized)
    # Create Corpus
    texts = data_lemmatized

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    return {
        'corpus': corpus,
        'id2word': id2word,
        'data_lemmatized': data_lemmatized
    }