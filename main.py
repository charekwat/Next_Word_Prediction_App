import streamlit as st
import shutil
import os

import numpy as np
from mailbox import ExternalClashError
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
import pickle
import numpy as np
import os
import pickle
import streamlit as st
from PIL import Image
import io

# Importing the Libraries

file = open("data1.txt", "r", encoding="utf8")
lines = []

for i in file:
    lines.append(i)

#print("The First Line: ", lines[0])
#print("The Last Line: ", lines[-1])
data = ""

for i in lines:
    data = ' '.join(lines)

data = data.replace('\n', '').replace('\r', '').replace('\ufeff', '')
#data[:400]
import string

translator = str.maketrans(string.punctuation, ' '*len(string.punctuation)) #map punctuation to space
new_data = data.translate(translator)

#new_data[:500]
q = []

for i in data.split():
    if i not in q:
        q.append(i)

data = ' '.join(q)
#data[:500]
tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])

# saving the tokenizer for predict function.
pickle.dump(tokenizer, open('tokenizer1.pkl', 'wb'))
sequence_data = tokenizer.texts_to_sequences([data])[0]
#sequence_data[:15]
sequences = []

for i in range(1, len(sequence_data)):
    words = sequence_data[i - 1:i + 1]
    sequences.append(words)

print("The Length of sequences are: ", len(sequences))
sequences = np.array(sequences)
#sequences[:5]
X = []
y = []

for i in sequences:
    X.append(i[0])
    y.append(i[1])

X = np.array(X)
y = np.array(y)

max_sequence_len = max([len(X) for x in sequences])
input_sequences = np.array(pad_sequences(sequences, maxlen=max_sequence_len, padding='pre'))
#input_sequences[1]
# Importing the Libraries

from tensorflow.keras.models import load_model
import numpy as np
import pickle

# Load the model and tokenizer

model = load_model('nextword1.h5')
tokenizer = pickle.load(open('tokenizer1.pkl', 'rb'))




def main():

    """Object detection App"""

    st.title("Next Word Prediction App")

    html_temp = """
    <body style="background-color:red;">
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Next word Prediction APP</h2>
    </div>
    </body>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    st.title(" Get the Predictions")
    seed_text = st.text_input("Enter a sentence of five words: ")

    if seed_text is not None:

        try:
            next_words = 1
            suggested_word = []
            # temp = seed_text
            for _ in range(next_words):

                token_list = tokenizer.texts_to_sequences([seed_text])[0]
                # print(token_list)
                token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
                predicted = np.argmax(model.predict(token_list), axis=-1)
                output_word = ""

                for word, index in tokenizer.word_index.items():
                    if index == predicted:
                        output_word = word
                        suggested_word.append(output_word)
                        break

                seed_text += " " + output_word
            print("Suggested next two word are : ", suggested_word)

           # print(seed_text)
        except Exception as e:
            print("Error occurred: ", e)


    if st.button("Suggested_two_words"):
        st.success(seed_text)


if __name__ == '__main__':
    main()

