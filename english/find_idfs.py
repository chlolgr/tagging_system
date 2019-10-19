import numpy as np
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import brown,reuters
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords,wordnet
from nltk import pos_tag,word_tokenize
import pickle as pkl
import time


news_text = list()
for category in brown.categories(): news_text+=list(brown.words(categories=category))
for category in reuters.categories(): news_text+=list(reuters.words(categories=category))

stopw = stopwords.words('english')
lemmer = WordNetLemmatizer()

all_text = ' '.join(news_text)
all_text = all_text.replace('-','')
all_text = all_text.replace(',','')

def pt_to_wn(tag):
	if tag.startswith('J'): return wordnet.ADJ
	if tag.startswith('V'): return wordnet.VERB
	if tag.startswith('N'): return wordnet.NOUN
	if tag.startswith('R'): return wordnet.ADV
	return wordnet.NOUN

words = list()
postags = pos_tag(word_tokenize(all_text))
for word,tag in postags: 
	if tag[0] in ['J','V','N','R']: words.append(lemmer.lemmatize(word,pt_to_wn(tag)))


unique_words,counts = np.unique(words,return_counts=True)

delete = np.where(counts<3)[0]
unique_words = np.delete(unique_words,delete)
counts = np.delete(counts,delete)

print('# of words in the corpora:',len(words))
print('# of distinct stems in the corpora:',len(unique_words))

#order = np.flip(np.argsort(counts),axis=0)
#for i in order[:20]: print(unique_words[i],counts[i])

idfs = dict(zip(unique_words,1/counts))

#pkl.dump(idfs,open('idfs_dict_lemm.pkl','wb'))
