# author：FLC
# time:2021/10/6
import pandas as pd


def load_embedding():
    path1 = '../data/glove.6B.100d.txt'
    glove_numpy = []
    glove_embedding_file = open(path1, encoding='utf-8')
    for line in glove_embedding_file:
        glove_line = line.strip().split(' ')
        glove_numpy.append(glove_line)
    glove_df = pd.DataFrame(glove_numpy)
    glove_df = glove_df.rename({0: 'word'}, axis='columns')
    path2 = '../data/crawl-300d-2M.vec'
    crawl_df = pd.read_csv(path2, sep=' ', header=None)
    crawl_df = crawl_df.drop(301, axis=1)
    crawl_df = crawl_df.rename({0: 'word'}, axis='columns')
    return glove_df, crawl_df

# load_embedding()

