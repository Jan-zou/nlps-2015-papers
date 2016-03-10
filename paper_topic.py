# !/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import pandas as pd
import numpy as np
import nltk
from gensim import corpora, models

def clean_text(text):
    list_of_cleaning_signs = ['\x0c', '\n']
    for sign in list_of_cleaning_signs:
        text = text.replace(sign, ' ')
    clean_text = re.sub('[^a-zA-Z]+', ' ', text)
    return clean_text.lower()


papers = pd.read_csv('output/Papers.csv')
papers['PaperText_clean'] = papers['PaperText'].apply(lambda x: clean_text(x))
clean_text = [text.split() for text in papers.PaperText_clean]

# bag of word
dictionary = corpora.Dictionary(clean_text)
corpus = [dictionary.doc2bow(word) for word in clean_text]
# tfidf
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]
# LDA
topic_num = 100
lda = models.ldamodel.LdaModel(corpus=corpus_tfidf, id2word=dictionary,
                               num_topics=topic_num, update_every=2, passes=10)

i = 0
for t in lda.print_topics(20):
    print "topic %s: " % i, t
    i += 1
