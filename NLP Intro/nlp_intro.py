"""
Created on Fri Nov  6 15:22:45 2020

@author: kkoni
"""
import pandas as pd
import nltk
import matplotlib.pyplot as plt
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords

# Text input and basic operations
tekst="W 1965 roku został członkiem Harcerskiego Klubu Taternickiego im. gen. Mariusza Zaruskiego w Katowicach. HKT, należący do Hufca Katowice-Zachód, prowadził działalność turystyczną na rzecz hufca, organizując imprezy turystyczne z wykorzystaniem elementów z zakresu wspinaczki. W działaniach tych brał czynny udział Jerzy Kukuczka. W 1966 roku wstąpił do Koła Katowickiego Klubu Wysokogórskiego i ukończył tatrzański kurs wspinaczkowy. W czasie harcerskiego zimowiska w Kowarach zimą 1967/68 uzyskał stopień przewodnika, a następnie – podharcmistrza. W HKT pełnił funkcje przewodniczącego Komisji Rewizyjnej (1967–1969) oraz szefa komórki szkoleniowo-kwalifikacyjnej (1971–1975). Razem z nim proporczyk Harcerskiego Klubu Taternickiego znalazł się na najwyższych szczytach Ziemi."
print(tekst[:20])
print(type(tekst))
print(len(tekst))

# Tokenization
sentences=sent_tokenize(tekst)
words=word_tokenize(tekst)
print("****Zdania")
#print(sentences)
print("****Słowa")
#print(words)

# Token frequency analysis
fdist=FreqDist(words)
print("****Top 20 tokenów")
print(fdist.most_common(20))
print(fdist.plot(10))

# Sanitize - limit to aplhanumeric only
words_sanitized=[]
for w in words:
    if w.isalpha():
        words_sanitized.append(w.lower())

fdist_sanitized=FreqDist(words_sanitized)
print("****Top 20 tokenów")
print(fdist_sanitized.most_common(20))
print(fdist_sanitized.plot(10))

# Remove stopwords
# stop=stopwords.words("english")
print(stop)

