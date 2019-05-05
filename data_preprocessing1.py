import nltk
import string
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences 


def load_data(data_path):
	global data
	data=pd.read_csv(data_path)
	#combine  title and text in one columns
	data['title']=data['title']+data['text']

	#delete unnecessary columns 
	data=data.drop(columns=['Unnamed: 0','text'])

	#data description
	print('-'*15,'Data Description','-'*15)
	print(data.head())
	print('\n')
	print('Number of  news in data : ',len(data['label']))
	print('Number of Fake news : ',(data['label']=='FAKE').sum())
	print('Number of REAL news : ',(data['label']=='REAL').sum())
	print('-'*48)

load_data('data.csv')
 

def data_clean(text):
	# Remove puncuation
	text = text.translate(string.punctuation)
	## Convert words to lower case and split them
	text = text.lower().split()
	## Remove stop words
	stops = set(stopwords.words("english"))
	text = [w for w in text if not w in stops and len(w) >= 3]
	text = " ".join(text)
	# Stemming
	text = text.split()
	stemmer = SnowballStemmer('english')
	stemmed_words = [stemmer.stem(word) for word in text]
	text = " ".join(stemmed_words)
	return text
# apply the above function to data['title']
data['title'] = data['title'].map(lambda x: data_clean(x))

data.to_csv('data_clean.csv')
