import random
import json
import pickle

import tensorflow as tf
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download('wordnet')

# Initialize lemmatizer and load intents file
lemmatizer = WordNetLemmatizer()
with open('intents.json') as file:
    intents = json.load(file)

words, classes, documents = [], [], []
ignore_letters = ['?', '!', '.', ',']

# Tokenize and lemmatize each pattern
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = sorted(set([lemmatizer.lemmatize(w) for w in words if w not in ignore_letters]))
classes = sorted(set(classes))

# Save words and classes
with open('words.pkl', 'wb') as file:
    pickle.dump(words, file)
with open('classes.pkl', 'wb') as file:
    pickle.dump(classes, file)

# Create training data
training_data = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = [1 if w in doc[0] else 0 for w in words]
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training_data.append(bag + output_row)

random.shuffle(training_data)
training_data = np.array(training_data)

x_train = training_data[:, :len(words)]
y_train = training_data[:, len(words):]

# Build and compile model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(len(x_train[0]),), activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(y_train[0]), activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
hist = model.fit(x_train, y_train, epochs=200, batch_size=5, verbose=1)

# Save the trained model
model.save('chatbot_model.keras', hist)
print('Model training completed and saved.')   