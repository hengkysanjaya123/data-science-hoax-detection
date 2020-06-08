# library to stem in Indonesian
# import Sastrawi

# library for stemming and tokenizing
from nltk.stem import PorterStemmer 
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
	ps = PorterStemmer()
	processed_text = ' '.join(ps.stem(token) for token in word_tokenize(text))
	token_list.append(processed_text)
	x = vectorizer.fit_transform(token_list)
	print(x)
	return x

#Example 
preprocess("GO go! makan dimakan go! going go go GO GO going")
