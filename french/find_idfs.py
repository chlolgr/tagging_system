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

def parse_xml(xml):
	xml = xml.split()
	writing = False
	words = list()
	for word in xml: 
		if word=='<p>': 
			writing = True
			continue
		if word=='</p>': 
			writing = False
			continue
		if writing: words.append(word)
	return ' '.join(words)

xml_filenames = glob.glob('Annee2003/*')+glob.glob('Annee2002/*')
all_text = ''
for i,xml in enumerate(xml_filenames[:150]): 
	print(i+1,'/',len(xml_filenames))
	all_text += parse_xml(codecs.open('Annee2003/2003-01-02.xml','r',errors='ignore').read())

#filename = 'wikipediaTXT.txt'
#all_text = codecs.open(filename,'r',encoding='latin-1',errors='ignore').read(20000000)
all_text = all_text.replace('-','')
all_text = all_text.replace(',','')
print('clean')
all_text = word_tokenize(all_text,language='french')
print('tokenized')

tagger = treetaggerwrapper.TreeTagger(TAGLANG='fr',TAGDIR=tagdir)
tags = tagger.tag_text(all_text)
print('tagged')

tags2 = treetaggerwrapper.make_tags(tags)
print('tagged again')

stopw = stopwords.words('french')

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
