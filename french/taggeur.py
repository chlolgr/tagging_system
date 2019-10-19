from nltk.corpus import stopwords
from nltk import word_tokenize
import treetaggerwrapper
#tagdir = '/home/chloe/Documents/snipfeed/prod/tagging_system/french/treetagger'
import numpy as np
import time
import re
import pickle as pkl
import unidecode
import spacy


def initialize_variables():
	idfs = pkl.load(open('/home/chloe/Documents/snipfeed/prod/tagging_system/french/idfs_dict_lemm.pkl','rb'))
	vocab = list(idfs.keys())
	len_vocab = len(vocab)
	word_to_int = dict(zip(vocab,range(len_vocab)))
	idfs_array = np.zeros(len_vocab)
	for word in vocab: idfs_array[word_to_int[word]] = idfs[word]
	stopw = stopwords.words('french')
	start = time.time()
	return vocab,len_vocab,word_to_int,idfs_array,stopw

def remove_unwanted(tags):
	remove = list()
	hour_re = '[0-2]?[0-9][h,:]([0-9]{2})?'
	days = ['lundi','mardi','mercredi','jeudi','vendredi','samedi','dimanche']
	for i,tag in enumerate(tags): 
		if not tag.isalnum(): 
			remove.append(tag)
			continue
		match = re.search(hour_re,tag)
		if match: 
			remove.append(tag)
			continue
		tags[i] = tag.replace('_','')
		#tags[i] = unidecode.unidecode(tag)
		if '|' in tag: remove.append(tag)
	for tag in remove: tags.remove(tag)
	return [tag for tag in tags if tag not in days and len(tag)>1]

def get_clean_text_raw(raw_text,stopw,complete=False):
	raw_text = raw_text.replace('-',' ')
	raw_text = raw_text.replace('’','\'')
	tagger = treetaggerwrapper.TreeTagger(TAGLANG='fr')
	tags = tagger.tag_text(raw_text)
	tags2 = treetaggerwrapper.make_tags(tags)
	if complete: postags = [(tag.word.lower(),tag.pos) for tag in tags2]
	else: postags = [(tag.lemma.lower(),tag.pos) for tag in tags2 if tag.pos[:3] in ['NOM','NAM','VER','ABR','ADJ'] and tag.word.lower() not in stopw]
	words = [word for word,tag in postags]
	postags = [tag for word,tag in postags]
	return words,postags

def get_TFs_raw(raw_text,len_vocab,word_to_int,stopw):
	words,postags = get_clean_text_raw(raw_text,stopw)
	NEs = np.array(words)[np.where(np.array(postags).astype(str)=='NAM')[0]]
	NE,counts = np.unique(NEs,return_counts=True)
	average = np.mean(counts)
	NEs = NE[np.where(counts>=average)[0]]
	word_counts = dict(zip(*np.unique(words,return_counts=True)))
	n_words = sum(list(word_counts.values()))
	tfs = np.zeros(len_vocab)
	unknowns = list()
	for i,word in enumerate(words): 
		try: tfs[word_to_int[word]] = word_counts[word]/n_words
		except KeyError: unknowns.append((word,postags[i]))
	return tfs,unknowns,NEs

def get_significants(raw_text,vocab,idfs_array,len_vocab,word_to_int,stopw):
	tfs,unknowns,NEs = get_TFs_raw(raw_text,len_vocab,word_to_int,stopw)
	tfidfs = tfs*idfs_array
	order = np.flip(np.argsort(tfidfs))
	significants = np.array(vocab)[order]
	return list(significants),list(tfidfs[order]),unknowns,list(NEs)

def get_sentences(raw_text,punctuation=False):
	raw_text = raw_text.replace('-',' ')
	raw_text = raw_text.replace('’','\'')
	sentences = list()
	sentence = ''
	for character in raw_text:
		if character in ['.','!','?']:
			sentences.append(sentence)
			sentence = ''
		else: sentence+=character
	for i,sentence in enumerate(sentences):
		if punctuation: sentences[i] = [word for word in word_tokenize(sentence,language='french')]
		else: sentences[i] = [word for word in word_tokenize(sentence,language='french') if word.isalnum() and not word.isdigit()]
	return sentences

def process_unknowns(unknowns):
	keeps = [word for word,tag in unknowns if tag[:3] in ['NOM','VER','NAM','ADJ'] and len(word)>=3]
	keeps = dict(zip(*np.unique(keeps,return_counts=True)))
	average = np.mean(list(keeps.values()))
	keeps = [keep for keep,count in keeps.items() if count>=average]
	return keeps
		
def get_spacy_significants(raw_text,vocab,idfs_array,len_vocab,word_to_int,stopw):
	significants,tfidfs,unknowns,NEs = get_significants(raw_text,vocab,idfs_array,len_vocab,word_to_int,stopw)
	tags = list()
	for i,s in enumerate(significants): 
		if tfidfs[i]<3e-5: break
		tags.append(s)
	tags += process_unknowns(unknowns)
	tags += NEs
	tags = remove_unwanted(tags)
	return list(set(tags))

def parse_tuples(raw_text,stopw):
	words,postags = get_clean_text_raw(raw_text,stopw,complete=True)
	tuples = list()
	for i in range(len(words)-1): 
		if ' '.join(postags[i:i+2]) not in ['NAM NAM','NOM ADJ','ADJ NOM','NOM NOM']: continue
		tuples.append('.'.join(words[i:i+2]))
	tuples,counts = np.unique(tuples,return_counts=True)
	average = np.mean(counts)
	return list(tuples[np.where(counts>=average)[0]])

def get_tags(raw_text):
	vocab,len_vocab,word_to_int,idfs_array,stopw = initialize_variables()
	tuples_ = parse_tuples(raw_text,stopw)
	doubles = list()
	for tuple_ in tuples_: doubles+=tuple_.split('.')
	tags = get_spacy_significants(raw_text,vocab,idfs_array,len_vocab,word_to_int,stopw)
	remove = list()
	for tag in tags: 
		if tag in doubles: 
			remove.append(tag)
			continue
	for tag in remove: tags.remove(tag)
	return tags+tuples_
