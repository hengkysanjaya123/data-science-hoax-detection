# library to stem in Indonesian
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# library for stemming and tokenizing
# from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize 
# library to generate TD-IDF
from sklearn.feature_extraction.text import TfidfVectorizer


# string input for text
# Removes prefixes, suffixes, punctuation 	
# Converts all text to lower case
# returns TF-IDF of the text
def stem(text):
    factory = StemmerFactory()
    ps = factory.create_stemmer()
    processed_text = ' '.join(ps.stem(token) for token in word_tokenize(text))
    return processed_text



#Example 
# preprocess("mereka pada dimakan para pemakan memakan untuk minum dan diminum setelah meminum")
    
print(word_tokenize('mereka pada dimakan para pemakan memakan untuk minum dan diminum setelah meminum'))
# mereka pada dimakan para pemakan memakan untuk minum dan diminum setelah meminum
# ['mereka', 'pada', 'dimakan', 'para', 'pemakan', 'memakan', 'untuk', 'minum', 'dan', 'diminum', 'setelah', 'meminum']