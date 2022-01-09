import string
import os
import math
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle

# Załaduje globalne stałe
nlp = spacy.load('en_core_web_sm')
punctuation = string.punctuation
stopwords = STOP_WORDS

# Funkcja preprocesująca maile. Zamienia stringa w liste nie-stop-wordowych tokenów
def preprocessor (msg: str):
    tokens = nlp(msg)
    tokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower() for word in tokens ]
    tokens = [ word for word in tokens if word not in stopwords and word not in punctuation ]
    return tokens

# Funkcja obliczająca wartość funkcji tfidf
def tfidf (term: str, document: list[str], corpus: list[list[str]]):
    return tf(term, document) * idf(term, corpus)

# Funkcje pomocnicze w obliczaniu tfidf
def tf (term: str, document: list[str]):
    return document.count(term) / len(document)

def idf (term: str, corpus: list[list[str]]):
    return math.log10(len(corpus) / (1 + len([doc for doc in corpus if term in doc])))

######## Część Główna ########

# Załaduj maile
data = pd.read_csv('./emails.csv', delimiter=',')

# Stwórz bazę do treningu i testów
train, test = train_test_split(data, train_size=0.9)

# Zresetuj indeksowanie w bazach. Niestety Pandas nie resetuje ich automatycznie.
train.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)

# Preprocessing danych
for n in range(0, len(train)):
    train.at[n, 'text'] = preprocessor(train.at[n, 'text'])
    if n%100 == 0:
        os.system('cls' if os.name=='nt' else 'clear')
        print("preprocessing data " + str(n) + "/" + str(len(train)))

# Zainicjowanie korpusów spamowych i niespamowych. Poprawia to nieco wyniki modelu.
spamCorpus = []
nonspamCorpus = []

# Skompletowanie korpusów
for n in range(0, len(train)):
    if train.at[n, "spam"] == 1:
        spamCorpus.append(set(train.at[n, "text"]))
    else:
        nonspamCorpus.append(set(train.at[n, "text"]))

# Zainicjowanie list, przechowujących wartości dla finalnej bazy
valuesSpam = []
valuesNonSpam = []
isSpam = []

# Skompletowanie wartości na ww. listach. Używam uśredniania wartości z tfidf ze względu na lepsze wyniki osiągane tą metodą.
for n in range(0, len(train)):
    valueSpam = 0
    valueNonSpam = 0
    doc = train.at[n, "text"]
    for token in doc:
        valueSpam += tfidf(token, doc, spamCorpus)
        valueNonSpam += tfidf(token, doc, nonspamCorpus)
    valueSpam = valueSpam / len(doc)
    valueNonSpam = valueNonSpam / len(doc)

    valuesSpam.append(valueSpam)
    valuesNonSpam.append(valueNonSpam)
    isSpam.append(train.at[n, "spam"])
    if n%100 == 0:
        os.system('cls' if os.name=='nt' else 'clear')
        print("processing data " + str(n) + "/" + str(len(train)))

# Utworzenie finalnej bazy
data_to_load = pd.DataFrame({'Spam Value': valuesSpam, 'NonSpam Value': valuesNonSpam, "Is Spam?": isSpam})

# Zwizualizuj dane na wykresie, zostanie wyświetlony po zakończeniu działania programu
data_to_load.plot.scatter(x='Spam Value',
                      y='NonSpam Value',
                      c='Is Spam?',
                      colormap='viridis')

# Podzielenie danych na cechy i wyniki
x = data_to_load.to_numpy()[:,0:2]
y = data_to_load.to_numpy()[:,2]

# Stworzenie modelu regresji logistycznej
clf = linear_model.LogisticRegression(C=1e40, solver='newton-cg')
fitted_model = clf.fit(x, y)

# Zainicjowanie zmiennych do oceny modelu
tp = 0
tn = 0
fp = 0
fn = 0

# Preprocesing danych testowych
for n in range(0, len(test)):
    test.at[n, 'text'] = preprocessor(test.at[n, 'text']) 
    if n%100 == 0:
        os.system('cls' if os.name=='nt' else 'clear')
        print("processing data " + str(n) + "/" + str(len(test)))

# Wyzerowanie list
valuesSpam = []
valuesNonSpam = []
isSpam = []

# Wyliczenie wartości do testowania
for n in range(0, len(test)):
    valueSpam = 0
    valueNonSpam = 0
    doc = test.at[n, "text"]
    for token in doc:
        valueSpam += tfidf(token, doc, spamCorpus)
        valueNonSpam += tfidf(token, doc, nonspamCorpus)
    valueSpam = valueSpam / len(doc)
    valueNonSpam = valueNonSpam / len(doc)

    valuesSpam.append(valueSpam)
    valuesNonSpam.append(valueNonSpam)
    isSpam.append(test.at[n, "spam"])

    if n%100 == 0:
        os.system('cls' if os.name=='nt' else 'clear')
        print("processing data " + str(n) + "/" + str(len(test)))

# Stworzenie bazy do testowania
data_to_test = pd.DataFrame({'Spam Value': valuesSpam, 'NonSpam Value': valuesNonSpam, "Is Spam?": isSpam})

# Testowanie modelu
for n in range(0, len(data_to_test)):
    if clf.predict([(data_to_test.at[n, "Spam Value"], data_to_test.at[n, "NonSpam Value"])]) == 0:
        if data_to_test.at[n, "Is Spam?"] == 0:
            tn += 1
        else:
            fn +=1
    else:
        if data_to_test.at[n, "Is Spam?"] == 1:
            tp += 1
        else:
            fp +=1

# Wyświetlenie wyników
print('Accuracy: ' + str((tp+tn)/(tp+tn+fp+fn)))
print('Precision: ' + str(tp/(tp+fp)))
print('Recall: ' + str(tp/(tp+fn)))
plt.show()

# Zapisanie modelu dla przyszłego użycia
filename = "model.sav"
pickle.dump(clf, open(filename, 'wb'))