import tensorflow as tf
import numpy as np
import random
import pickle

random.seed(random.randint(0, 10000000))

with open('data.txt', 'r') as file:
    data = file.read()

tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True)
tokenizer.fit_on_texts([data])

max_id = len(tokenizer.word_index)
[encoded] = np.array(tokenizer.texts_to_sequences([data])) - 1
dataset_size = len(encoded)
train_size = int(dataset_size * 0.90)
dataset = tf.data.Dataset.from_tensor_slices(encoded[:train_size])

eval_dataset = tf.data.Dataset.from_tensor_slices(encoded[train_size:])

print(max_id)
print(tokenizer.sequences_to_texts([[i for i in range(max_id)]]))

embedding_size = 4
batch_size = 256
n_steps = 100
window_length = n_steps + 1

dataset = dataset.window(window_length, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(window_length))
dataset = dataset.shuffle(10000, seed=random.randint(0, 100000)).batch(batch_size)
dataset = dataset.map(lambda windows: (windows[:, :-1], windows[:, 1:]))
dataset = dataset.prefetch(2)

eval_dataset = eval_dataset.window(window_length, shift=1, drop_remainder=True)
eval_dataset = eval_dataset.flat_map(lambda window: window.batch(window_length))
eval_dataset = eval_dataset.shuffle(10000, seed=random.randint(0, 100000)).batch(batch_size)
eval_dataset = eval_dataset.map(lambda windows: (windows[:, :-1], windows[:, 1:]))
eval_dataset = eval_dataset.prefetch(2)


with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

save_cb = tf.keras.callbacks.ModelCheckpoint('twitchRNNSave1e3.h5', save_freq=100)


model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(max_id, embedding_size),
    tf.keras.layers.GRU(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
    tf.keras.layers.GRU(256, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(max_id, activation='softmax'))
])
model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

model = tf.keras.models.load_model('twitchRNNSave1e3.h5')


# history = model.fit(dataset, epochs=50, callbacks=[save_cb], validation_freq=1, validation_data=eval_dataset, use_multiprocessing=True, workers=8)


def preprocess(texts):
    X = np.array(tokenizer.texts_to_sequences(texts)) - 1
    return X


def next_char(text, temperature=1.0):
    x_new = preprocess([text])
    y_proba = model.predict(x_new)[0, -1:, :]
    char_id = tf.argmax(y_proba)
    return tokenizer.sequences_to_texts([char_id.numpy()])[0]


def complete_text(text, n_chars=100, temperature=1.0):
    for _ in range(n_chars):
        text += next_char(text, temperature)
        if text[-1] == '\n':
            return text
    return text


X_new = preprocess(['How are yo'])
Y_pred = model(X_new)
print(Y_pred)


def predict():
    info = ''
    for i in range(50):
        print(i)
        rand = random.randint(0, len(data) - 150)
        start = data[rand:rand + 150] + '\n'
        info += complete_text(start, temperature=random.uniform(0.75, 1.25)) + '\n\n'

    info = info.replace(chr(0xe0000), '')
    info = info.replace(chr(0x1fad8), '')
    info = info.replace(chr(0x1fae1), '')

    with open('chats.txt', 'w') as file:
        file.write(info)


predict()
