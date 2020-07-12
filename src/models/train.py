import json
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

def train():
    with open(r"../data/interim/corpus.pkl", "rb") as input_file:
        corpus = pickle.load(input_file)

    with open(r"../data/interim/id2word.pkl", "rb") as input_file:
        id2word = pickle.load(input_file)

    with open(r"../data/interim/data_lemmatized.pkl", "rb") as input_file:
        data_lemmatized = pickle.load(input_file)

    lda_model = gensim.models.ldamodel.LdaModel(
        corpus=corpus,
        id2word=id2word,
        num_topics=10, 
        random_state=100,
        update_every=1,
        chunksize=100,
        passes=10,
        alpha='auto',
        per_word_topics=True
    )

    # Save on a pickel the model
    with open(r"../data/interim/lda_model.pkl", "wb") as output_file:
        pickle.dump(lda_model, output_file)

    doc_lda = lda_model[corpus]

    # Save topics into json file
    topics = lda_model.show_topics(formatted=False)
    p_topics = []
    for i, probs in topics:
        p_topic = {
            'index': i,
            'words': []
        }
        for w, prob in probs:
            p_topic['words'].append({
                'word': w,
                'prob': prob.astype(float)
            })
        p_topics.append(p_topic)
    
    with open("../data/interim/topics.json", "w") as output_file:
        json.dump(p_topics, output_file)

    # Score de coherencia
    # coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
    # coherence_lda = coherence_model_lda.get_coherence()

train()