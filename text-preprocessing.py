# library to stem in Indonesian
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# library for stemming and tokenizing
# from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize 
# library to generate TD-IDF
from sklearn.feature_extraction.text import TfidfVectorizer


# string input for text
# Removes prefixes, suffixes, punctuation 	
# Converts all text to lower case
# returns TF-IDF of the text
def preprocess(text):
	token_list = []
	vectorizer = TfidfVectorizer()
	factory = StemmerFactory()
	ps = factory.create_stemmer()
	processed_text = ' '.join(ps.stem(token) for token in word_tokenize(text))
	token_list.append(processed_text)
	x = vectorizer.fit_transform(token_list)
	print(token_list)
	print(x)
	return x

#Example 
preprocess("dimakan makan memakan")
