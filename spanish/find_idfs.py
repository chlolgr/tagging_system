import numpy as np
from nltk.corpus import stopwords,wordnet
from nltk import pos_tag,word_tokenize
import pickle as pkl
import codecs
import treetaggerwrapper
tagdir = '/home/chloe/Documents/snipfeed/prod/tagging_system/french/treetagger'
from pprint import pprint
import glob
import time

text_file = open('text.txt','r')
text = text_file.read()

text = word_tokenize(text,language='spanish')


tagger = treetaggerwrapper.TreeTagger(TAGLANG='es',TAGDIR=tagdir)
tags = tagger.tag_text(text)
print('tagged')


tags2 = treetaggerwrapper.make_tags(tags)
print('tagged again')

stopw = stopwords.words('spanish')

words = list()
for tag in tags2: 
	try: 
		if tag.lemma.lower() not in stopw and tag.lemma.isalnum(): words.append(tag.lemma)
	except AttributeError: pass


unique_words,counts = np.unique(words,return_counts=True)

delete = np.where(counts<3)[0]
unique_words = np.delete(unique_words,delete)
counts = np.delete(counts,delete)

print('# of words in the corpora:',len(words))
print('# of distinct stems in the corpora:',len(unique_words))

order = np.flip(np.argsort(counts),axis=0)
for i in order[:20]: print(unique_words[i],counts[i])

idfs = dict(zip(unique_words,1/counts))

pkl.dump(idfs,open('idfs_dict_lemm.pkl','wb'))
