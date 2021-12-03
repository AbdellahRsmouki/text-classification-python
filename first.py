# https://www.slideshare.net/commercetools/boosting-product-categorization-with-machine-learning?from_action=save
# https://www.youtube.com/watch?v=4Hh0RNdKKKs
import io
import json
from tensorflow.keras.layers import Input
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Embedding,LSTM,Dense,SpatialDropout1D
from tensorflow.keras.applications.mobilenet import MobileNet
from nltk.tokenize import word_tokenize
import gensim
import pandas as pd
from nltk.corpus import stopwords
import nltk
# nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('averaged_perceptron_tagger')

stop_english=set(stopwords.words('english'))
import os
import random
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
#import cv2
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
from tensorflow.compat.v1.keras.initializers import Constant
from sklearn.model_selection import train_test_split





data = pd.read_csv('train.csv')
#pid = list(data['ImgId'])
#descriptions = list(data['description'])
data.head()



data['categories'].value_counts().plot(kind='bar', figsize=(10, 5));

def depth(field, n, sep=' > '):
    """Split category depth helper"""
    if n <= 0:
        return field
    return sep.join(field.split(sep, n)[: n])

# Category depth
default_depth = 2

# Min n of samples per category
min_samples = 50


word_len=data['description'].str.len()
# plt.hist(word_len, bins=50)
# plt.ylabel('Samples')
# plt.xlabel('Description Character Length')
# plt.title('Characters in the description')
# plt.show()



# Text preprocessing¶

def get_token(description):
        # split the description into tokens (words)
        tokens = set(gensim.utils.tokenize(description))
        # Avoid words does not have atleast 2 character 
        tokens = [i for i in tokens if(len(i) > 2)]
        # Remove stop words
        tokens = [s for s in tokens if s not in stop_english]
        return tokens

for product in data:
    if product.get('category') and any([
        product.get('description'),
        product.get('features'),
        product.get('name')
    ]):
        if depth(
                product.get('category'), default_depth
        ) in products.get('target_names'):

            if categories.get(depth(product.get('category'), default_depth)):

                products['target'].append(
                    products.get('target_names').index(
                        depth(product.get('category'), default_depth)
                    )
                )

                products['data'].append(
                    u'{}. {}. {}'.format(
                        product.get('description') or '',
                        product.get('features') or '',
                        product.get('name') or ''
                    )
                )

with io.open('products.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(data, ensure_ascii=False))






