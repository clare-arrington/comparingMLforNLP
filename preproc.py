import re
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

#%%
from nltk.corpus import wordnet as wn

stops = set(stopwords.words("english"))

def is_verb(tag):
    return tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
def is_adverb(tag):
    return tag in ['RB', 'RBR', 'RBS']
def is_adjective(tag):
    return tag in ['JJ', 'JJR', 'JJS']
def penn_to_wn(tag):
    if is_adjective(tag):
        return wn.ADJ
    elif is_adverb(tag):
        return wn.ADV
    elif is_verb(tag):
        return wn.VERB
    return wn.NOUN

#%%

wine = pd.read_csv('winemag-data-130k-v2.csv', index_col='id')
wine = wine.drop_duplicates().reset_index()
wine['binary_label'] = np.where(wine.points < 90, 0, 1)
wine = wine[['points','binary_label','desc']]
wine.desc = [re.sub('\s?\d+','',re.sub(r'[^\w\s]','',aWine.lower())) for aWine in wine.desc]

#%%
lemmdesc = []
times = []
lmtzr = WordNetLemmatizer()

for i in range(0, len(wine)):
    sentence = []
    for tag in nltk.pos_tag(nltk.word_tokenize(wine.desc[i])):
        sentence.append(lmtzr.lemmatize(tag[0],penn_to_wn(tag[1])))
    lemmdesc.append(" ".join(sentence))
    if i % 10000 == 0:
        print(i)
        print(str(round((i / 120000) * 10)) + '/12 Completed')
    elif (i == len(wine) - 1):
        print('Done!')
        
wine.desc = lemmdesc 
wine.to_csv('wine.csv', index=False) 
