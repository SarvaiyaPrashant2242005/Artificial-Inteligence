import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer

sentenses = [
    "I love my Dog",
    "I love my cat",
    "Yoy love my Dog!",
    "Do you think my Dog is amazing"
]

tokenizer = Tokenizer(num_words = 100)
tokenizer.fit_on_texts(sentenses)

word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(sentenses)
print(word_index)
print(sequences)