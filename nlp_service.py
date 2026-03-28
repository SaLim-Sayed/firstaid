#!/usr/bin/env python
# coding: utf-8


import tensorflow.keras as keras
import numpy as np
import pandas as pd
import pickle
import random
import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()
model_path = 'firstaid_model.h5'
model = keras.models.load_model(model_path)
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
intents = pd.read_json('intents.json', encoding = 'utf-8-sig')

class _nlp_service:
    model = None
    _instance = None
    
    def get_response(self, audio_text):
        #  make prediction
        ints = self.predict_class(audio_text)
        try:
            tag = ints[0]['intent']
            list_of_intents = intents['intents']
            for i in list_of_intents:
                if i['tag']  == tag:
                    response = random.choice(i['responses'])
                    break
        except IndexError:
            response = "I don't understand!"
        return response   

        
        
    def preprocess(self, audio_text, words):
        sentence_words = nltk.word_tokenize(audio_text)
        sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        bag = [0] * len(words)
        for s in sentence_words:
            for i, word in enumerate(words):
                if word == s:
                    bag[i] = 1
        return np.array(bag)
    
    
    def predict_class(self, audio_text):
        bag_of_words = self.preprocess(audio_text, words)
        res = model.predict(np.array([bag_of_words]))[0]
        ERROR_THRESHOLD = 0.5
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
        return return_list
        
    
def nlp_service():
    # ensure that we only have 1 instance of nlp-model
    if _nlp_service._instance is None:
        _nlp_service._instance = _nlp_service()
        _nlp_service.model = keras.models.load_model(model_path)
    return _nlp_service._instance

if __name__=='__main__':
    response = nlp_service()
    response = response.get_response('i have fever')
    
    print(response)



