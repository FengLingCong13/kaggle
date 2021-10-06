# author：FLC
# time:2021/10/6
import numpy as np
import pandas as pd
import csv

def load_embedding():

    path1 = '../data/glove.6B.100d.txt'
    glove_words = []
    glove_embedding = []
    glove_embedding_file = open(path1, encoding='utf-8')
    for line in glove_embedding_file:
        glove_line = line.strip().split(' ')
        glove_words.append(glove_line[0])
        glove_vector = [t for t in glove_line[1:]]
        glove_embedding.append(glove_vector)
    glove_words = np.array(glove_words)
    glove_embedding = np.array(glove_embedding)

    path2 = '../data/crawl-300d-2M.vec'
    craw_words = []
    craw_embedding = []
    craw_embedding_file = open(path2, encoding='utf-8')
    n = 0;
    for line in craw_embedding_file:
        if n==0:
            n+=1
            continue
        craw_line = line.strip().split(' ')
        craw_words.append(craw_line[0])
        craw_vector = [t for t in craw_line[1:]]
        craw_embedding.append(craw_vector)
    craw_words = np.array(craw_words)
    craw_embedding = np.array(craw_embedding)
    print(craw_words.shape)
    print(craw_embedding.shape)
    return glove_words, glove_embedding, craw_words, craw_embedding

load_embedding()

