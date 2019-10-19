from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
from english import tagger
from french import taggeur
from spanish import el_taggor

en_stopwords = list(stopwords.words('english'))
es_stopwords = list(stopwords.words('spanish'))
fr_stopwords = list(stopwords.words('french'))

def detect_language(text):
	words = word_tokenize(text)
	languages = ['en','es','fr']
	scores = np.zeros(3)
	for word in words: 
		if word in en_stopwords: scores[0]+=1
		if word in es_stopwords: scores[1]+=1
		if word in fr_stopwords: scores[2]+=1
	return languages[np.argmax(scores)]


def select_tagger(text):
	language = detect_language(text)
	if language=='en': return tagger.get_tags(text)
	if language=='fr': return taggeur.get_tags(text)
	if language=='es': return el_taggor.get_tags(text)



if __name__=='__main__':

	text = """
	"I've done what I came to do," Rooney told Fox News. He said he ran for Congress to "get the money for the Everglades projects that had been languishing for many years, and to try to get this offshore drilling ban passed to protect Florida."
Rooney, who won his first election in 2016, said he initially thought his goals would take three terms, "but I think I've done it in less than two."
Francis Rooney is the rare House Republican open to impeaching Trump
Asked by Fox News if he needed or wanted to pursue a third term in office, Rooney said, "I don't really think I do, and I don't really think I want one."
The congressman said he wanted to be a "model for term limits," and added: "People need to realize ... this is public service not public life."
Rooney -- a member of the House Foreign Affairs Committee, which is at the center of the impeachment inquiry into Trump -- said Friday he had not yet come to a conclusion on whether the President committed a crime that compels his removal from office. His statement was a striking one among House Republicans defensive of Trump.
MORE ON THE IMPEACHMENT INQUIRY FROM CNN
Trump impeachment inquiry: A visual timeline

The congressman said Mick Mulvaney, the acting White House chief of staff, confirmed Thursday what Trump had denied -- that the President engaged in a quid pro quo with Ukraine. Rooney also said he was eager to learn from the witnesses coming in next week.
Mulvaney told reporters the Trump administration "held up the money" for Ukraine because Trump wanted to investigate "corruption" in Ukraine related to a conspiracy theory involving the whereabouts of the Democratic National Committee's computer server hacked by Russians during the 2016 presidential campaign. When pressed on whether the President sought an exchange of favors, Mulvaney said, "We do that all the time with foreign policy."
Rooney said some Republicans might be afraid of being rebuked by the party if they expressed skepticism about the President, saying, "It might be the end of things for me...depending on how things go."
But, he said, "I didn't take this job to keep it."
	"""

	tags = select_tagger(text)
	print(tags)
