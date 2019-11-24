import tensorflow as tf
from tensorflow import keras
import numpy as np

data = keras.datasets.imdb

# load data
(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=10000)

# modify data
word_index = data.get_word_index()

word_index = {k:(v+3) for k, v in word_index.items()}
# special case word items
# <PAD> --> add padding to the end of the review to fill a maxlen
# <START> --> indicate the start of the review
# <UNK> --> unknown words
# <UNUSED> --> unused words
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)

# convert integer coded text into string words
def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])

# add model
model = keras.Sequential()
model.add(keras.layers.Embedding(10000, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))

model.summary()

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# split validation and test set from train data and labels
x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]

# fit model
fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)

results = model.evaluate(test_data, test_labels)

print(results)

'''
# prediction for the first test review
test_review = test_data[0]
predict = model.predict([test_review])
print("Review: ")
print(decode_review(test_review))
print("Prediction: " + str(predict[0]))
print("Actual: " + str(test_labels[0]))
print(results)
'''