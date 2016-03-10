# !/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


def given_paperID_give_index(paper_id, paper_data):
    return paper_data[paper_data['Id']==paper_id].index[0]


if __name__ == '__main__':
    papers = pd.read_csv('output/Papers.csv')

    tfidf_vectorizer_PaperText = TfidfVectorizer(ngram_range=(2,2), stop_words="english",
                                    max_df=0.9, min_df=0.1, max_features=2000)
    tfidf_matrix_PaperText = tfidf_vectorizer_PaperText.fit_transform(papers.PaperText)
    terms_PaperText = tfidf_vectorizer_PaperText.get_feature_names()

    num_neighbors = 6
    nbrs = NearestNeighbors(n_neighbors=num_neighbors).fit(tfidf_matrix_PaperText)
    distances_PaperText, indices_PaperText = nbrs.kneighbors(tfidf_matrix_PaperText)

    # eg
    Ex_paper_id = 5941
    Ex_index = given_paperID_give_index(Ex_paper_id, papers)
    print ("The Text of the example paper is:\n")
    print (papers.iloc[indices_PaperText[Ex_index][0]]['PaperText'])
    print ("The Text of the similar papers are:\n")
    for i in range(1, len(indices_PaperText[Ex_index])):
        print ("Neighbor No. %r has following text: \n" % i)
        print (papers.iloc[indices_PaperText[Ex_index][i]]['PaperText'])
        print ("\n")
