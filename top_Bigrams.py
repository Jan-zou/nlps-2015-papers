# !/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

EXCLUDED_BIGRAMS = [
    "et al",
    "10 10",
    "international conference",
    "neural information",
    "information processing",
    "processing systems",
    "advances neural",
    "supplementary material"
]

def get_top_bigrams(inputfile):
    papers = pd.read_csv(inputfile)

    cv = CountVectorizer(ngram_range=(2, 2), stop_words="english", max_features=500)
    cv.fit(papers.PaperText)
    X = cv.transform(papers.PaperText)
    counts = X.sum(axis=0)    # add each col

    df = pd.DataFrame({'Bigrams': cv.get_feature_names(),
                       'Counts': counts.tolist()[0]})
    df = df[df.Bigrams.map(lambda x: x not in EXCLUDED_BIGRAMS)]
    # inplace: change raw DataFrame
    df.sort_values(by='Counts', ascending=False, inplace=True)

    return df


if __name__ == '__main__':
    df = get_top_bigrams("output/Papers.csv")
    print df
