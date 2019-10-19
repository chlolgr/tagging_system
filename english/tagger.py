from nltk.corpus import stopwords,wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk import word_tokenize,pos_tag
import numpy as np
import time
import re
import pickle as pkl
import spacy

# RETURNS ALL NEEDED VARIABLES
#	vocab 		:: all known words (list)
#	len_vocab 	:: number of known words (int)
#	word_to_int	:: dictionary from a word to its index in the vocab (dict)
#	idfs_array	:: inverse document frequencies of all words in vocab (numpy array)
#	stopw		:: english stopwords (list)
#	porter		:: instance of PorterStemmer (PorterStemmer)
#	nlp		:: tool for nlp tasks from the spacy library
def initialize_variables():
	idfs = pkl.load(open('/home/chloe/Documents/snipfeed/prod/tagging_system/english/idfs_dict_lemm.pkl','rb'))
	vocab = list(idfs.keys())
	len_vocab = len(vocab)
	word_to_int = dict(zip(vocab,range(len_vocab)))
	idfs_array = np.zeros(len_vocab)
	for word in vocab: idfs_array[word_to_int[word]] = idfs[word]
	stopw = stopwords.words('english')
	lemmer = WordNetLemmatizer()
	nlp = spacy.load('en_core_web_sm')
	return vocab,len_vocab,word_to_int,idfs_array,stopw,lemmer,nlp

# CONVERTS PTB POSTAGS TO WORNET TYPES
def pt_to_wn(tag):
	if tag.startswith('J'): return wordnet.ADJ
	if tag.startswith('V'): return wordnet.VERB
	if tag.startswith('N'): return wordnet.NOUN
	if tag.startswith('R'): return wordnet.ADV

# REMOVES ALL UNWANTED TAGS FROM THE FINAL LIST
# 	days of the week (ex: 'monday')
#	days of the month (ex: '16th')
# 	times (ex: '6:35pm')
#	ownership 's
#	tags of one letter
def remove_unwanted(tags):
	remove = list()
	days = ['monday','tuesday','wednesday','thursday','friday','saturday','sunday']
	day_of_the_month_re = '[1-9][1-9]?th'
	time_re = '[0-1]?[0-9]:[0-5][0-9]([a,p]m)?'
	belonging_re = '\.\'s'
	for i,tag in enumerate(tags): 
		match = re.search(belonging_re,tag)
		if match:
			start,end = match.span()
			tags[i] = tag[:start]+tag[end:]
		for bit in tag.split('.'):
			match = re.search(day_of_the_month_re,bit)
			if match: 
				remove.append(tag)
				break
			match = re.search(time_re,bit)
			if match: remove.append(tag)
	for tag in remove: tags.remove(tag)
	return [tag for tag in tags if tag not in days and len(tag)>1]

# TAKES IN RAW TEXT AND RETURNS CLEAN TEXT FOR TF-IDF PURPOSES
#	keeps nouns, adjectives, verbs, foreign words
#	stems the remaining words and sets them to lowercase
def get_clean_text_raw(raw_text,stopw,lemmer):
	text = word_tokenize(raw_text)
	postags = pos_tag(text)
	postags = [(word,tag) for word,tag in postags if tag in ['NN','JJ','JJR','NNS','VB','VBD','VBG','VBN','VBP','VBZ']]
	text = [lemmer.lemmatize(word.lower(),pt_to_wn(tag)) for word,tag in postags if word.lower() not in stopw and word.isalnum() and not word.isdigit()]
	postags = [tag for word,tag in postags if word.lower() not in stopw and word.isalnum() and not word.isdigit()]
	return text,postags

# TAKES IN RAW TEXT AND RETURNS TFS + UNKNOWN WORDS
#	all known words in TF vector size of len_vocab
#	unknown words saved to use as tags
def get_TFs_raw(raw_text,len_vocab,word_to_int,stopw,lemmer):
	words,postags = get_clean_text_raw(raw_text,stopw,lemmer)
	word_counts = dict(zip(*np.unique(words,return_counts=True)))
	n_words = sum(list(word_counts.values()))
	tfs = np.zeros(len_vocab)
	unknowns = list()
	for i,word in enumerate(words): 
		try: tfs[word_to_int[word]] = word_counts[word]/n_words
		except KeyError: unknowns.append((word,postags[i]))
	return tfs,unknowns

# TAKES IN RAW TEXT + OUR VARIABLES AND RETURNS TAGS FROM TF-IDF
#	returns all known words in order of tf-idf + tf-idfs in reverse order + unknown words, separately
def get_significants(raw_text,vocab,idfs_array,len_vocab,word_to_int,stopw,lemmer):
	tfs,unknowns = get_TFs_raw(raw_text,len_vocab,word_to_int,stopw,lemmer)
	tfidfs = tfs*idfs_array
	order = np.flip(np.argsort(tfidfs))
	significants = np.array(vocab)[order]
	return list(significants),list(tfidfs[order]),unknowns

def get_significants_from_title(raw_title,word_to_int,lemmer):
	ptags = pos_tag(word_tokenize(raw_title))
	keeps = list()
	for word,tag in ptags:
		if tag.startswith('N') and word.isalnum() and not word.isdigit():
			word = lemmer.lemmatize(word.lower())
			try: 
				word_to_int[word]
				keeps.append(word)
			except KeyError: pass
	return keeps

# TAKES IN RAW TEXT AND RETURNS LIST OF LISTS OF WORDS (one list per sentence)
#	if punctuation set to True -> punctuation returned with words
#	punctuation set to False by default -> only words returned
def get_sentences(raw_text,punctuation=False):
	raw_text = raw_text.replace('-','')
	sentences = list()
	sentence = ''
	for character in raw_text:
		if character in ['.','!','?']:
			sentences.append(sentence)
			sentence = ''
		else: sentence+=character
	for i,sentence in enumerate(sentences):
		if punctuation: sentences[i] = [word for word in word_tokenize(sentence)]
		else: sentences[i] = [word for word in word_tokenize(sentence) if word.isalnum() and not word.isdigit()]
	return sentences

# TAKES IN TWO LISTS OF WORDS AND RETURNS TRUE IF THEY HAVE ONE IN COMMON, FALSE OTHERWISE
def common_words(entity1,entity2):
	words1 = entity1.split()
	words2 = entity2.split()
	for word in words1:
		if word in words2: return True
	return False

# PROCESSES NAMED ENTITIES THAT ARE EITHER PEOPLE OR ORGANIZATIONS
#	only keeps one instance of each person
#	counts number of times each person is referred to
#	keeps only people who appear more than average
#	keeps every organization named (filtered for unicity at the end)
def process_ppl_org(entities,len_text):
	counts = dict() # store number of times we've seen a person's name
	cleaned_entities = list() # list we'll return at the end
	for entity in entities: 
		add = True # whether to add it to the list at the end
		if entity.label_=='ORG':
			cleaned_entities.append(entity.text)
			continue
		if entity.label_=='PERSON': 
			for r in counts: 
				if common_words(r,entity.text): 
					# we've already seen that person's name
					counts[r]+=1
					add=False
					if len(entity.text.split())>len(r.split()): 
						# this instance of the name is longer (full name VS last name for instance)
						# so we update the key in the dictionary
						counts[entity.text] = counts[r]
						# and delete the old key
						del counts[r]
					break
			# if we get to this point it's the first time we see that name
			# now realizing 'add' has become useless with updates
			# too scared to remove it on github
			# too lazy to remove it in my py script and test it on local
			if add: counts[entity.text] = 1
	if len(counts)>0:
		# we have at least one person
		average_appearance = np.mean(list(counts.values())) # average number of times a person is mentioned
		all_words = list()
		for entity,count in counts.items():
			add=True
			if count>=average_appearance: # only keep people who appear more than average
				cleaned_entities.append(entity)
	for i,entity in enumerate(cleaned_entities): 
		# remove anything that isn't a word (it happened)
		cleaned_entities[i] = ' '.join([word for word in entity.split() if word.isalnum()])
	return cleaned_entities

# PROCESSES NAMED ENTITIES THAT ARE PLACES
def process_places(places,len_text,stopw):
	keeps = list()
	average_count = np.mean(list(places.values()))
	for place,counts in places.items(): 
		if counts>=average_count: 
			keeps.append('.'.join([word.lower() for word in place.split() if word.lower() not in stopw]))
	return keeps

# TAKES IN RAW TEXT + OUR VARIABLES AND RETURNS NAMED ENTITIES DETECTED WITH SPACY
def get_spacy_entities(raw_text,stopw,nlp,lemmer):
	sentences = get_sentences(raw_text,punctuation=True)
	len_text = sum([len(sentence) for sentence in sentences])
	entities = list()
	places = dict()
	for sentence in sentences: 
		sentence = ' '.join(sentence)
		doc = nlp(sentence)
		for entity in doc.ents:
			if entity.label_ in ['PERSON','ORG'] and len(entity.text.split())<=2: entities.append(entity)
			if entity.label_=='GPE': 
				try: places[entity.text] += 1
				except KeyError: places[entity.text] = 1
	entities = process_ppl_org(entities,len_text)
	for i,entity in enumerate(entities):
		entities[i] = '.'.join([word.lower() for word in entity.split() if word.lower() not in stopw])
	places = process_places(places,len_text,stopw)
	return entities+places

def process_unknowns(unknowns):
	porter = PorterStemmer()
	keeps = list()
	stems = dict()
	for word,tag in unknowns:
		stem = porter.stem(word)
		try: stems[stem].append((word,tag))
		except KeyError: stems[stem] = [(word,tag)]
	for stem,appearances in stems.items():
		if len(appearances)>1:
			done = False
			for word,tag in appearances:
				if tag=='NN': 
					keeps.append(word)
					done = True
					break
			if done: continue
			for word,tag in appearances:
				if tag.startswith('N'): 
					keeps.append(word)
					done = True
					break
			if done: continue
			for word,tag in appearances:
				if tag=='JJ': 
					keeps.append(word)
					done = True
					break
			if done: continue
			for word,tag in appearances:
				if tag.startswith('J'): 
					keeps.append(word)
					done = True
					break
			if done: continue
			for word,tag in appearances: 
				if tag.startswith('V'): 
					keeps.append(word)
					break
	return keeps

def get_spacy_significants(raw_text,lemmer,vocab,idfs_array,len_vocab,word_to_int,stopw,nlp):
	entities = get_spacy_entities(raw_text,stopw,nlp,lemmer)
	doubles = list()
	for entity in entities: doubles += entity.split('.')
	doubles = [tag for tag in doubles]
	significants,tfidfs,unknowns = get_significants(raw_text,vocab,idfs_array,len_vocab,word_to_int,stopw,lemmer)
	tags = list()
	for i,s in enumerate(significants): 
		if tfidfs[i]<.0005: break
		if s not in doubles: tags.append(s)
	tags += process_unknowns(unknowns)
	tags += entities
	return list(set(tags))

def get_tags(raw_text,raw_title=None):
	vocab,len_vocab,word_to_int,idfs_array,stopw,lemmer,nlp = initialize_variables()
	tags = get_spacy_significants(raw_text,lemmer,vocab,idfs_array,len_vocab,word_to_int,stopw,nlp)
	if raw_title: 
		title_tags = get_significants_from_title(raw_title,word_to_int,lemmer)
		tags = list(set(title_tags+tags))
	tags = remove_unwanted(tags)
	return tags



if __name__=='__main__': 

	from newspaper import Article

	def open_link(link):
		article = Article(link)
		article.download()
		article.parse()
		return article.text,article.title

	link = "https://www.foxnews.com/entertainment/queen-elizabeth-notre-dame-fire-saddened"
	text,title = open_link(link)

	start = time.time()
	print('#'+'\n#'.join(get_tags(text,title)))
	print('\ntime:',time.time()-start)
