import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
def load_data(path):
	global df
	df = pd.read_csv(path)
	df.drop('Unnamed: 0', axis=1, inplace=True)
	df['title'] = df['title'].str.replace(r'http[\w:/\.]+','<URL>')
	df['title'] = df['title'].str.replace(r'[^\.\w\s]','') #remove everything but characters and punctuatio
	df['title'] = df['title'].str.replace(r'\.\.+','.') #replace multple periods with a single one
	df['title'] = df['title'].str.replace(r'\.',' . ') #replace periods with a single one
	df['title'] = df['title'].str.replace(r'\s\s+',' ') #replace multple white space with a single one
	df['title'] = df['title'].str.strip() 
	print(df.shape)
	print(df.head())
load_data('data_clean.csv')



with open('glove.6B/glove.6B.200d.txt','rb') as f:
    lines = f.readlines()

glove_weights = np.zeros((len(lines), 200))
words = []
for i, line in enumerate(lines):
    word_weights = line.split()
    words.append(word_weights[0])
    weight = word_weights[1:]
    glove_weights[i] = np.array([float(w) for w in weight])
word_vocab = [w.decode("utf-8") for w in words]

word2glove = dict(zip(word_vocab, glove_weights))

all_text = ' '.join(df.title.values)
words = all_text.split()



u_words = Counter(words).most_common()
u_words_counter = u_words
u_words_frequent = [word[0] for word in u_words if word[1]>10] # we will only consider words that have been used more than 10 times

u_words_total = [k for k,v in u_words_counter]
word_vocab = dict(zip(word_vocab, range(len(word_vocab))))
word_in_glove = np.array([w in word_vocab for w in u_words_total])

words_in_glove = [w for w,is_true in zip(u_words_total,word_in_glove) if is_true]
words_not_in_glove = [w for w,is_true in zip(u_words_total,word_in_glove) if not is_true]

# # create the dictionary
word2num = dict(zip(words_in_glove,range(len(words_in_glove))))
len_glove_words = len(word2num)
freq_words_not_glove = [w for w in words_not_in_glove if w in u_words_frequent]
b = dict(zip(freq_words_not_glove,range(len(word2num), len(word2num)+len(freq_words_not_glove))))
word2num = dict(**word2num, **b)
word2num['<Other>'] = len(word2num)
num2word = dict(zip(word2num.values(), word2num.keys()))

int_text = [[word2num[word] if word in word2num else word2num['<Other>'] 
             for word in content.split()] for content in df.title.values]

# plt.plot([len(t) for t in int_text])
# plt.xlabel('News Articles')
# plt.ylabel('Words Number')
# plt.show()

num2word[len(word2num)] = '<PAD>'
word2num['<PAD>'] = len(word2num)

for i, t in enumerate(int_text):
    if len(t)<3000:
        int_text[i] = [word2num['<PAD>']]*(3000-len(t)) + t
    elif len(t)>3000:
        int_text[i] = t[:3000]
    else:
        continue

x = np.array(int_text)
y = (df['label'].map({'FAKE':0,'REAL':1}))

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)

model = keras.models.Sequential()
model.add(keras.layers.Embedding(len(word2num), 200))
# model.add(keras.layers.LSTM(100))
# model.add(keras.layers.Dropout(0.5))
# model.add(keras.layers.Dense(100, activation=tf.nn.relu))
#-----------------------------------------------------------

# model.add(keras.layers.GRU(100))
# model.add(keras.layers.Dropout(0.5))
# model.add(keras.layers.Dense(100, activation=tf.nn.relu))

#---------------------------------------------------------
# model.add(keras.layers.Dropout(0.5))
# model.add(keras.layers.GlobalAveragePooling1D())
# model.add(keras.layers.Dense(200, activation=tf.nn.relu))
#-----------------------------------------------------------
# model.add(keras.layers.Conv1D(filters=200,kernel_size=8,activation='relu'))
# model.add(keras.layers.MaxPooling1D(pool_size=4))
# model.add(keras.layers.Dropout(0.5))
# model.add(keras.layers.LSTM(100))
# model.add(keras.layers.Dropout(0.5))
# model.add(keras.layers.Dense(100, activation=tf.nn.relu))
#-----------------------------------------------------------
model.add(keras.layers.Conv1D(filters=200,kernel_size=8,activation='relu'))
model.add(keras.layers.MaxPooling1D(pool_size=4))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.GRU(100))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(100, activation=tf.nn.relu))
# -----------------------------------------------------------


model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
#rmsprop
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
earlystopper = keras.callbacks.EarlyStopping(monitor='val_loss', verbose=1)

history=model.fit(X_train, y_train,callbacks=[earlystopper], batch_size=256, epochs=100, validation_split=0.3, verbose=1)
model.save('Conv1D.h5')  # creates a HDF5 file 'my_model.h5'

history_dict = history.history
history_dict.keys()

acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)
plt.plot(epochs, loss, 'b', label='Conv1D Training loss')
plt.plot(epochs, val_loss, 'r-', label='Conv1D Validation loss')

plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.clf()   # clear figure
plt.plot(epochs, acc, 'b', label='Conv1D Training acc')
plt.plot(epochs, val_acc, 'r-', label='Conv1D Validation acc')

plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
results = model.evaluate(X_test, y_test)
print(results)
