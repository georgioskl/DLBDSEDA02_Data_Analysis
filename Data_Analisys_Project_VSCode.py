import re
import pandas as pd

import nltk.corpus
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation

df = pd.read_csv('consumer_complaints.csv', skipfooter= 551757, engine='python') ## skippen 551757 daten und soll nur die 4000 lesen 


## Bereinigung des Beswerdetextes.
df['issue'] = df['issue'].str.lower() ## Kleinbuchstabenumwandlung.

df['issue'] = df['issue'].apply(word_tokenize) ## Tokenisierung.

stop_words = set(stopwords.words("english"))
df['issue'] = df['issue'].apply(lambda text: [token for token in text if token not in stop_words]) ## Entfernung von Stoppwörtern.

stemmer = SnowballStemmer("english")
df['issue'] = df['issue'].apply(lambda tokens: [stemmer.stem(token) for token in tokens]) ## Stemming.

lemma = WordNetLemmatizer()
df['issue'] = df['issue'].apply(lambda tokens: [lemma.lemmatize(token) for token in tokens]) ## Lemmatisierung.


complaints = []
for row in df['issue']:
    complaints.append(row) ## Die Beschwerden sind so strukturiert, dass sie dazu dienen, den Wortschatz zu erweitern.

results = [' '.join(ele) for ele in df['issue']]
complaints = ' '.join(results)
complaints = word_tokenize(complaints) ## wird für BoW mit Sklearn verwendet.


voc = []
for w in complaints:
    if w not in voc:
        voc.append(w)
print(voc) ## Der Wortschatz wurde aufgebaut, wobei jedes einzelne bereinigte Wort aus den Beschwerden nur einmal im Vokabular erscheint.


def CalcBow(voc, complaints):
    return {word: complaints.count(word) for word in voc} 
dataframe_Bow = pd.DataFrame([CalcBow(voc, r) for r in df['issue']])
print(dataframe_Bow) ##Der Wortschatz wurde aufgebaut, wobei jedes einzelne bereinigte Wort aus den Beschwerden nur einmal im Vokabular erscheint.


vectorizer1 = CountVectorizer()
data = vectorizer1.fit_transform(results)
data = pd.DataFrame(data.toarray(), columns=vectorizer1.get_feature_names_out())
print(data.head()) ### Erstellung BoW mit der verwendung von sklearn.

vectorizer2 = TfidfVectorizer(min_df=1)
mod = vectorizer2.fit_transform(results)
data_IDF = pd.DataFrame(mod.toarray(), columns=vectorizer2.get_feature_names_out())
print(data_IDF.head()) ## TF-IDF


## Semantische Analyse
lda_mod1 = LatentDirichletAllocation(n_components=7, learning_method='online', random_state=42)
lda = lda_mod1.fit_transform(data)
print("Topic: ")
for i, topic in enumerate(lda[0]):
    print("Topic: ",i,": ",topic*100,"%") ## LDA-Algorithmus


voca1 = vectorizer1.get_feature_names_out()
num_topics = 7

for i, comp in enumerate(lda_mod1.components_):
    sd_words = sorted(zip(voca1, comp), key=lambda x: x[1], reverse=True)[:10]

print(f"Topic {i}:")
print(' '.join(word for word, _ in sd_words))
print() ## Die Schlüsselwörter für jedes Thema.

lsa_mod2 = TruncatedSVD(n_components=7, algorithm='randomized', n_iter=10)
lsa = lsa_mod2.fit_transform(mod)
print("Topic: ")
for i, topic in enumerate(lsa[0]):
    print("Topic: ",i,": ",topic*100,"%") ## LSA-Algorithmus


voca2 = vectorizer2.get_feature_names_out()
num_topics = 7

for i, comp in enumerate(lsa_mod2.components_):
    sd_words = sorted(zip(voca2, comp), key=lambda x: x[1], reverse=True)[:10]

print(f"Topic {i}:")
print(' '.join(word for word, _ in sd_words))
print() ## Die Schlüsselwörter für jedes Thema.