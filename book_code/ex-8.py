'''Sentiment analysis example with IMDB from p. 211 of the book.'''

from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense
from keras.datasets import imdb

'Runs p. 211 sentinment analysis network.'
def main():
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=5000)
    vocab_size = max(max(X_train))

    max_review_len = max([len(seq) for seq in X_train] + [len(seq) for seq in X_test])
    X_train = pad_sequences(X_train, maxlen=max_review_len)
    X_test = pad_sequences(X_test, maxlen=max_review_len)


    model = Sequential()
    model.add(Embedding(vocab_size, 100, input_length=max_review_len))
    model.add(LSTM(256))
    model.add(Dense(512))
    model.add(Dense(512))
    model.add(Dense(512))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64)
    result = model.evaluate(X_test, y_test, verbose=0)

main()
