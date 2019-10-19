from nltk.corpus import stopwords,wordnet
from nltk import word_tokenize,pos_tag
import numpy as np
import time
import re
import pickle as pkl
import spacy
import treetaggerwrapper
tagdir = '/home/chloe/Documents/snipfeed/prod/tagging_system/french/treetagger'
from newspaper import Article


def initialize_variables():
    idfs = pkl.load(open('idfs_dict_lemm.pkl','rb'))
    vocab = list(idfs.keys())
    len_vocab = len(vocab)
    word_to_int = dict(zip(vocab,range(len_vocab)))
    idfs_array = np.zeros(len_vocab)
    for word in vocab: idfs_array[word_to_int[word]] = idfs[word]
    stopw = stopwords.words('spanish')
    nlp = spacy.load('es_core_news_sm')
    return vocab,len_vocab,word_to_int,idfs_array,stopw,nlp

def remove_unwanted(tags):
    remove = list()
    days = ['lunes','martes','miércoles','jueves','viernes','sábado','domingo']
    return [tag for tag in tags if tag not in days and len(tag)>1]

def get_clean_text_raw(raw_text,stopw):
    words,postags = list(),list()
    sentences = get_sentences(raw_text,punctuation=True)
    tagger = treetaggerwrapper.TreeTagger(TAGLANG='es',TAGDIR=tagdir)
    tags = tagger.tag_text(text)
    tags2 = treetaggerwrapper.make_tags(tags)
    for tag in tags2:
        if tag.lemma.lower() not in stopw and ((tag.pos.startswith('N') and tag.pos!='NP') or tag.pos.startswith('V') or tag.pos in ['ADJ','ADV']): 
            words.append(tag.lemma.lower())
            postags.append(tag.pos)
    return words,postags

def get_TFs_raw(raw_text,len_vocab,word_to_int,stopw,nlp):
    words,postags = get_clean_text_raw(raw_text,stopw)
    word_counts = dict(zip(*np.unique(words,return_counts=True)))
    n_words = sum(list(word_counts.values()))
    tfs = np.zeros(len_vocab)
    unknowns = list()
    for i,word in enumerate(words): 
        try: tfs[word_to_int[word]] = word_counts[word]/n_words
        except KeyError: unknowns.append((word,postags[i]))
    return tfs,unknowns

def get_significants(raw_text,vocab,idfs_array,len_vocab,word_to_int,stopw,nlp):
    tfs,unknowns = get_TFs_raw(raw_text,len_vocab,word_to_int,stopw,nlp)
    tfidfs = tfs*idfs_array
    order = np.flip(np.argsort(tfidfs))
    significants = np.array(vocab)[order]
    return list(significants),list(tfidfs[order]),unknowns

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
        if punctuation: sentences[i] = [word for word in word_tokenize(sentence,language='spanish')]
        else: sentences[i] = [word for word in word_tokenize(sentence,language='spanish') if word.isalnum() and not word.isdigit()]
    return sentences

def common_words(entity1,entity2):
    words1 = entity1.split()
    words2 = entity2.split()
    for word in words1:
        if word in words2: return True
    return False

def process_ppl_org(entities,len_text):
    counts = dict()
    cleaned_entities = list()
    for entity in entities: 
        add = True
        if entity.label_=='ORG':
            cleaned_entities.append(entity.text)
            continue
        if entity.label_=='PER': 
            for r in counts: 
                if common_words(r,entity.text): 
                    counts[r]+=1
                    add=False
                    if len(entity.text.split())>len(r.split()): 
                        counts[entity.text] = counts[r]
                        del counts[r]
                    break
            if add: counts[entity.text] = 1
    if len(counts)>0:
        average_appearance = np.mean(list(counts.values()))
        all_words = list()
        for entity,count in counts.items():
            if count>=average_appearance: 
                cleaned_entities.append(entity)
    for i,entity in enumerate(cleaned_entities): 
        cleaned_entities[i] = ' '.join([word for word in entity.split() if word.isalnum()])
    return cleaned_entities

def process_places(places,len_text,stopw):
    keeps = list()
    average_count = np.mean(list(places.values()))
    for place,counts in places.items(): 
        if counts>=average_count: 
            keeps.append('.'.join([word.lower() for word in place.split() if word.lower() not in stopw]))
    return keeps

def get_spacy_entities(raw_text,stopw,nlp):
    sentences = get_sentences(raw_text,punctuation=True)
    len_text = sum([len(sentence) for sentence in sentences])
    entities = list()
    places = dict()
    for sentence in sentences: 
        sentence = ' '.join(sentence)
        doc = nlp(sentence)
        for entity in doc.ents:
            if entity.label_ in ['PER','ORG'] and len(entity.text.split())<=2: entities.append(entity)
            if entity.label_ in ['GPE','LOC']: 
                try: places[entity.text] += 1
                except KeyError: places[entity.text] = 1
    entities = process_ppl_org(entities,len_text)
    for i,entity in enumerate(entities):
        entities[i] = '.'.join([word.lower() for word in entity.split() if word.lower() not in stopw])
    places = process_places(places,len_text,stopw)
    return entities+places

def process_unknowns(unknowns):
    keeps = dict()
    for word,tag in unknowns:
        try: keeps[word] += 1
        except KeyError: keeps[word] = 1
    average = np.mean(list(keeps.values()))
    return [word for word,count in keeps.items() if count>=average and word.isalpha()]

def get_spacy_significants(raw_text,vocab,idfs_array,len_vocab,word_to_int,stopw,nlp):
    entities = get_spacy_entities(raw_text,stopw,nlp)
    doubles = list()
    for entity in entities: doubles += entity.split('.')
    doubles = [tag for tag in doubles]
    significants,tfidfs,unknowns = get_significants(raw_text,vocab,idfs_array,len_vocab,word_to_int,stopw,nlp)
    tags = list()
    for i,s in enumerate(significants): 
        if tfidfs[i]<.0005: break
        if s not in doubles: tags.append(s)
    tags += [word for word in process_unknowns(unknowns) if word not in doubles]
    tags += entities
    tags = remove_unwanted(tags)
    return list(set(tags))

def get_tags(raw_text):
    raw_text = raw_text.replace('¡','')
    vocab,len_vocab,word_to_int,idfs_array,stopw,nlp = initialize_variables()
    return get_spacy_significants(raw_text,vocab,idfs_array,len_vocab,word_to_int,stopw,nlp)
