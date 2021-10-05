# author：FLC
# time:2021/10/5
import string

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn


def input_data():
    path_train = '../data/train.csv'
    path_test = '../data/test.csv'
    path_stopwords = '../data/stopwords.txt'
    df_train = pd.read_csv(path_train)
    df_test = pd.read_csv(path_test)
    df_stopwords = pd.read_csv(path_stopwords,header=None)
    return df_train, df_test, df_stopwords


def count_missing_values(df_train, df_test):
    train_missing_keyword = df_train['keyword'].isnull().sum()
    train_missing_location = df_train['location'].isnull().sum()
    test_missing_keyword = df_test['keyword'].isnull().sum()
    test_missing_location = df_test['location'].isnull().sum()

    x = ['keyword', 'location']
    fig, ax = plt.subplots(1, 2)
    ax[0].bar(x, [train_missing_keyword, train_missing_location], color=['blue', 'orange'])
    ax[1].bar(x, [test_missing_keyword, test_missing_location], color=['blue', 'orange'])
    ax[0].set_title('train_data')
    ax[1].set_title('test_data')
    ax[0].set_ylabel('Missing Value')
    ax[1].set_ylabel('Missing Value')
    plt.tight_layout()
    plt.show()
    print('train_missing_keyword_per:', round(train_missing_keyword / df_train.shape[0] * 100, 2), '%')
    print('train_missing_location_per:', round(test_missing_location / df_train.shape[0] * 100, 2), '%')
    print('test_missing_keyword_per:', round(train_missing_keyword / df_train.shape[0] * 100, 2), '%')
    print('test_missing_location_per:', round(test_missing_location / df_train.shape[0] * 100, 2), '%')


def count_unique(df_train, df_test):
    train_unique_keyword = df_train['keyword'].nunique()
    test_unique_keyword = df_train['keyword'].nunique()

    train_unique_location = df_train['location'].nunique()
    test_unique_location = df_test['location'].nunique()

    print('unique_keyworkd:', train_unique_keyword, '(train)  ', test_unique_keyword, '(test)')
    print('unique_location:', train_unique_location, '(train)  ', test_unique_location, '(test)')


def meta_fetures_statistics(df_train, df_test, df_stopwords):
    word_count_train = df_train['text'].apply(lambda x: len(str(x).split())).values.reshape((-1, 1))
    word_count_test = df_test['text'].apply(lambda x: len(str(x).split())).values.reshape((-1, 1))

    unique_word_count_train = df_train['text'].apply(lambda x: np.unique(str(x).split()).shape[0]).values.reshape(
        (-1, 1))
    unique_word_count_test = df_test['text'].apply(lambda x: np.unique(str(x).split()).shape[0]).values.reshape((-1, 1))

    stop_word_count_train = df_train['text'].apply(
        lambda x: len([w for w in str(x).lower().split() if w in df_stopwords[0].tolist()])).values.reshape((-1, 1))
    stop_word_count_test = df_test['text'].apply(
        lambda x: len([w for w in str(x).lower().split() if w in df_stopwords[0].tolist()])).values.reshape((-1, 1))

    url_count_train = df_train['text'].apply(
        lambda x: len([w for w in str(x).lower().split() if 'http' in w or 'https' in w or 'www' in w])).values.reshape((-1, 1))
    url_count_test = df_test['text'].apply(
        lambda x: len([w for w in str(x).lower().split() if 'http' in w or 'https' in w or 'www' in w])).values.reshape((-1, 1))

    mean_word_length_train = df_train['text'].apply(lambda x: np.mean([len(w) for w in str(x).split()])).values.reshape(
        (-1, 1))
    mean_word_length_test = df_test['text'].apply(lambda x: np.mean([len(w) for w in str(x).split()])).values.reshape(
        (-1, 1))

    char_count_train = df_train['text'].apply(lambda x: len(str(x))).values.reshape((-1, 1))
    char_count_test = df_test['text'].apply(lambda x: len(str(x))).values.reshape((-1, 1))

    punctuation_count_train = df_train['text'].apply(
        lambda x: len([c for c in str(x) if c in string.punctuation])).values.reshape((-1, 1))
    punctuation_count_test = df_test['text'].apply(
        lambda x: len([c for c in str(x) if c in string.punctuation])).values.reshape((-1, 1))

    hashtag_count_train = df_train['text'].apply(lambda x: len([c for c in str(x) if c == '#'])).values.reshape(
        (-1, 1))
    hashtag_count_test = df_test['text'].apply(lambda x: len([c for c in str(x) if c == '#'])).values.reshape(
        (-1, 1))

    mention_count_train = df_train['text'].apply(lambda x: len([c for c in str(x) if c == '@'])).values.reshape(
        (-1, 1))
    mention_count_test = df_test['text'].apply(lambda x: len([c for c in str(x) if c == '@'])).values.reshape(
        (-1, 1))
    train_statistics = np.concatenate((word_count_train, unique_word_count_train,
                                       stop_word_count_train, url_count_train,
                                       mean_word_length_train, char_count_train,
                                       punctuation_count_train, hashtag_count_train,
                                       mention_count_train,df_train['target'].values.reshape((-1,1))), axis=1)
    test_statistics = np.concatenate((word_count_test, unique_word_count_test,
                                      stop_word_count_test,
                                      url_count_test, mean_word_length_test, char_count_test,
                                      punctuation_count_test, hashtag_count_test,
                                      mention_count_test), axis=1)

    train_statistics = pd.DataFrame(train_statistics, columns=['word_count_train', 'unique_word_count_train',
                                                               'stop_word_count_train', 'url_count_train',
                                                               'mean_word_length_train', 'char_count_train',
                                                               'punctuation_count_train', 'hashtag_count_train',
                                                               'mention_count_train','train_target'],dtype=np.int32)

    test_statistics = pd.DataFrame(test_statistics, columns=['word_count_test', 'unique_word_count_test',
                                                             'stop_word_count_test',
                                                             'url_count_test', 'mean_word_length_test',
                                                             'char_count_test',
                                                             'punctuation_count_test', 'hashtag_count_test',
                                                             'mention_count_test'],dtype=np.int32)
    return train_statistics, test_statistics


def plot_meta_fetures_statistics(train_statistics, test_statistics):
    fig, ax = plt.subplots(9, 2, figsize=(20, 50))

    seaborn.distplot(train_statistics['word_count_train'],ax=ax[0][1], label='train')
    seaborn.distplot(test_statistics['word_count_test'],ax=ax[0][1], label='test')

    seaborn.distplot(train_statistics['unique_word_count_train'], ax=ax[1][1], label='train')
    seaborn.distplot(test_statistics['unique_word_count_test'], ax=ax[1][1], label='test')

    seaborn.distplot(train_statistics['stop_word_count_train'], ax=ax[2][1], label='train')
    seaborn.distplot(test_statistics['stop_word_count_test'], ax=ax[2][1], label='test')

    seaborn.distplot(train_statistics['url_count_train'], ax=ax[3][1], label='train')
    seaborn.distplot(test_statistics['url_count_test'], ax=ax[3][1], label='test')

    seaborn.distplot(train_statistics['mean_word_length_train'], ax=ax[4][1], label='train')
    seaborn.distplot(test_statistics['mean_word_length_test'], ax=ax[4][1], label='test')

    seaborn.distplot(train_statistics['char_count_train'], ax=ax[5][1], label='train')
    seaborn.distplot(test_statistics['char_count_test'], ax=ax[5][1], label='test')

    seaborn.distplot(train_statistics['punctuation_count_train'], ax=ax[6][1], label='train')
    seaborn.distplot(test_statistics['punctuation_count_test'], ax=ax[6][1], label='test')

    seaborn.distplot(train_statistics['hashtag_count_train'], ax=ax[7][1], label='train')
    seaborn.distplot(test_statistics['hashtag_count_test'], ax=ax[7][1], label='test')

    seaborn.distplot(train_statistics['mention_count_train'], ax=ax[8][1], label='train')
    seaborn.distplot(test_statistics['mention_count_test'], ax=ax[8][1], label='test')

    TWEETS = train_statistics['train_target'] == 1
    NOT_TWEETS = train_statistics['train_target'] == 0


    seaborn.distplot(train_statistics.loc[TWEETS]['word_count_train'], ax=ax[0][0], label='Not Disaster')
    seaborn.distplot(train_statistics.loc[NOT_TWEETS]['word_count_train'], ax=ax[0][0], label='Disaster')
    ax[0][0].legend()
    ax[0][0].set_title('word_count_train Distribution')
    ax[0][0].set_xlabel(' ')
    ax[0][1].legend()
    ax[0][1].set_title('word_count Distribution')
    ax[0][1].set_xlabel(' ')

    seaborn.distplot(train_statistics.loc[TWEETS]['unique_word_count_train'], ax=ax[1][0], label='Not Disaster')
    seaborn.distplot(train_statistics.loc[NOT_TWEETS]['unique_word_count_train'], ax=ax[1][0], label='Disaster')
    ax[1][0].legend()
    ax[1][0].set_title('unique_word_count_train Distribution')
    ax[1][0].set_xlabel(' ')
    ax[1][1].legend()
    ax[1][1].set_title('unique_word_count Distribution')
    ax[1][1].set_xlabel(' ')

    seaborn.distplot(train_statistics.loc[TWEETS]['stop_word_count_train'], ax=ax[2][0], label='Not Disaster')
    seaborn.distplot(train_statistics.loc[NOT_TWEETS]['stop_word_count_train'], ax=ax[2][0], label='Disaster')
    ax[2][0].legend()
    ax[2][0].set_title('stop_word_count_train Distribution')
    ax[2][0].set_xlabel(' ')
    ax[2][1].legend()
    ax[2][1].set_title('stop_word_count Distribution')
    ax[2][1].set_xlabel(' ')

    seaborn.distplot(train_statistics.loc[TWEETS]['url_count_train'], ax=ax[3][0], label='Not Disaster')
    seaborn.distplot(train_statistics.loc[NOT_TWEETS]['url_count_train'], ax=ax[3][0], label='Disaster')
    ax[3][0].legend()
    ax[3][0].set_title('url_count_train Distribution')
    ax[3][0].set_xlabel(' ')
    ax[3][1].legend()
    ax[3][1].set_title('url_count Distribution')
    ax[3][1].set_xlabel(' ')

    seaborn.distplot(train_statistics.loc[TWEETS]['mean_word_length_train'], ax=ax[4][0], label='Not Disaster')
    seaborn.distplot(train_statistics.loc[NOT_TWEETS]['mean_word_length_train'], ax=ax[4][0], label='Disaster')
    ax[4][0].legend()
    ax[4][0].set_title('mean_word_length_train Distribution')
    ax[4][0].set_xlabel(' ')
    ax[4][1].legend()
    ax[4][1].set_title('mean_word_length Distribution')
    ax[4][1].set_xlabel(' ')

    seaborn.distplot(train_statistics.loc[TWEETS]['char_count_train'], ax=ax[5][0], label='Not Disaster')
    seaborn.distplot(train_statistics.loc[NOT_TWEETS]['char_count_train'], ax=ax[5][0], label='Disaster')
    ax[5][0].legend()
    ax[5][0].set_title('char_count_train Distribution')
    ax[5][0].set_xlabel(' ')
    ax[5][1].legend()
    ax[5][1].set_title('char_count Distribution')
    ax[5][1].set_xlabel(' ')

    seaborn.distplot(train_statistics.loc[TWEETS]['punctuation_count_train'], ax=ax[6][0], label='Not Disaster')
    seaborn.distplot(train_statistics.loc[NOT_TWEETS]['punctuation_count_train'], ax=ax[6][0], label='Disaster')
    ax[6][0].legend()
    ax[6][0].set_title('punctuation_count_train Distribution')
    ax[6][0].set_xlabel(' ')
    ax[6][1].legend()
    ax[6][1].set_title('punctuation_count Distribution')
    ax[6][1].set_xlabel(' ')

    seaborn.distplot(train_statistics.loc[TWEETS]['hashtag_count_train'], ax=ax[7][0], label='Not Disaster')
    seaborn.distplot(train_statistics.loc[NOT_TWEETS]['hashtag_count_train'], ax=ax[7][0], label='Disaster')
    ax[7][0].legend()
    ax[7][0].set_title('hashtag_count_train Distribution')
    ax[7][0].set_xlabel(' ')
    ax[7][1].legend()
    ax[7][1].set_title('hashtag_count Distribution')
    ax[7][1].set_xlabel(' ')

    seaborn.distplot(train_statistics.loc[TWEETS]['mention_count_train'], ax=ax[8][0], label='Not Disaster')
    seaborn.distplot(train_statistics.loc[NOT_TWEETS]['mention_count_train'], ax=ax[8][0], label='Disaster')
    ax[8][0].legend()
    ax[8][0].set_title('mention_count_train Distribution')
    ax[8][0].set_xlabel(' ')
    ax[8][1].legend()
    ax[8][1].set_title('mention_count Distribution')
    ax[8][1].set_xlabel(' ')

    plt.show()

df_train, df_test, df_stopwords = input_data()
# count_missing_values(df_train, df_test)
# count_unique(df_train, df_test)
train_statistics, test_statistics = meta_fetures_statistics(df_train, df_test, df_stopwords)
plot_meta_fetures_statistics(train_statistics, test_statistics)

