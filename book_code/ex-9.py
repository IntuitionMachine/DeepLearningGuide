'''Natural language processing example with Karpathy's char-rnn using LSTM nodes from p. 224 of the book.

Uses dataset from https://einstein.ai/research/the-wikitext-long-term-dependency-language-modeling-dataset'''

from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.utils import to_categorical
import numpy as np


def main():
    wiki_data = "ex-9-data\wikitext-2\wiki.train.tokens"
    max_length = 5

    vocab_max = 5000
    tokenizer = Tokenizer(num_words=vocab_max)
    tokenizer.fit_on_texts([wiki_data])
    trainset = tokenizer.texts_to_sequences([wiki_data])[0]
    vocab_size = len(tokenizer.word_index) + 1
    cluster_size = 6
    sequences = []
    for i in range(cluster_size-1, len(trainset)):
        sequence = trainset[i-cluster_size+1:i+1]
        sequences.append(sequence)
    sequences = np.array(sequences)
    x_sequences, y = sequences[:,:-1], sequences[:,-1]
    y = to_categorical(y, num_classes=vocab_size)

    model = Sequential()
    model.add(Embedding(vocab_size, 200, input_length=max_length-1))
    model.add(LSTM(256))
    model.add(Dense(512))
    model.add(Dense(512))
    model.add(Dense(512))
    model.add(Dense(vocab_size, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, y, epochs=50)
    # the seed txt and n_words are user-configurable
    seed_text = ""
    n_words = 1
    generate_sample_text(model, tokenizer, seed_text, n_words)

def generate_sample_text(model, tokenizer, seed_text, n_words):
    input_text = seed_text
    final_output = ""
    for _ in range(n_words):
        input_sequence = tokenizer.texts_to_sequences([input_text])[0]
        y_index = model.predict_classes(np.array(encoded))
        for word, index in tokenizer.word_index.items():
            if index == y_index:
                output = word
                break
        input_text += ' ' + output
        final_output += output
    print(seed_text, " -> ",final_output)

main()
