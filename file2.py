import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

class ChatBot:
    def __init__(self, intents_file, words_file, classes_file, model_file):
        self.lemmatizer = WordNetLemmatizer()
        with open(intents_file) as file:
            self.intents = json.load(file)
        self.words = pickle.load(open(words_file, 'rb'))
        self.classes = pickle.load(open(classes_file, 'rb'))
        self.model = load_model(model_file)
    
    def clean_up_sentence(self, sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [self.lemmatizer.lemmatize(word) for word in sentence_words]
        return sentence_words
    
    def bag_of_words(self, sentence):
        sentence_words = self.clean_up_sentence(sentence)
        bag = [0] * len(self.words)
        for w in sentence_words:
            if w in self.words:
                bag[self.words.index(w)] = 1
        return np.array(bag)
    
    def predict_class(self, sentence):
        bow = self.bag_of_words(sentence)
        res = self.model.predict(np.array([bow]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = [{'intent': self.classes[r[0]], 'probability': str(r[1])} for r in results]
        return return_list
    
    def get_response(self, intents_list):
        tag = intents_list[0]['intent']
        list_of_intents = self.intents['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                return random.choice(i['responses'])
        return "Sorry, I didn't understand that."

def main():
    bot = ChatBot('intents.json', 'words.pkl', 'classes.pkl', 'chatbot_model.keras')
    print("GO! Bot is running!")
    while True:
        message = input("")
        ints = bot.predict_class(message)
        res = bot.get_response(ints)
        print(res)

if __name__ == "__main__":
    main()    