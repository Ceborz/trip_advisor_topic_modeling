import pickle
import json
from typing import List
from pprint import pprint

from data.data_utils import clean_strings
from features.utils import get_corpus_info


def predict(documents: List[str]):
    # Load the model
    with open(r"../data/interim/lda_model.pkl", "rb") as input_file:
        lda_model = pickle.load(input_file)

    if not lda_model:
        return {
            'status': 'Error'
        }
    
    # Load interpreted topics
    with open("../data/interim/interpreted_topics.json", "r") as input_file:
        topics = json.load(input_file)
    
    # Prepare inputs
    documents = clean_strings(documents)

    predictions = []
    for document in documents:
        corpus_info = get_corpus_info([document])
        corpus = corpus_info['corpus']
        doc_lda = lda_model[corpus]
        index = -1
        max_prob = -1
        for i, prob in doc_lda[0][0]:
            if prob > max_prob:
                max_prob = prob
                index = i

        topic = lda_model.show_topics(index, formatted=False)
        topic_words = []
        for idx, sub_topic in topic:
            for w in sub_topic:
                topic_words.append(w)
        
        index = str(index)
        topic = 'NA'
        if index in topics:
            topic = topics[index]

        predictions.append({
            'document': document,
            'result': {
                'topic': topic,
                'index': index
            }
        })

    return predictions

# Load file
with open('../data/test/test_prediction.txt', "r") as input_file:
    lines = input_file.readlines()

documents = [x.strip() for x in lines]
pprint (predict(documents))