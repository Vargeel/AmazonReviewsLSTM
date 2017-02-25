"""
This script is what created the dataset pickled.

1) You need to download this file and put it in the same directory as this file.
https://github.com/moses-smt/mosesdecoder/raw/master/scripts/tokenizer/tokenizer.perl . Give it execution permission.

2) Get the dataset from http://ai.stanford.edu/~amaas/data/sentiment/ and extract it in the current directory.

3) Then run this script.
"""

dataset_path='data/'

import numpy as np
import cPickle as pkl
import six.moves.cPickle as pickle
import os
from subprocess import Popen, PIPE

# tokenizer.perl is from Moses: https://github.com/moses-smt/mosesdecoder/tree/master/scripts/tokenizer
tokenizer_cmd = ['./tokenizer.perl', '-l', 'en', '-q', '-']


def tokenize(sentences):

    print 'Tokenizing..',
    text = "\n".join(sentences)
    tokenizer = Popen(tokenizer_cmd, stdin=PIPE, stdout=PIPE, shell=True)
    tok_text, _ = tokenizer.communicate(text)
    toks = tok_text.split('\n')[:-1]
    print 'Done'

    return toks


def build_dict(sentences):

    currdir = os.getcwd()
    os.chdir(currdir)

    sentences = tokenize(sentences)

    print 'Building dictionary..',
    wordcount = dict()
    for ss in sentences:
        words = ss.strip().lower().split()
        for w in words:
            if w not in wordcount:
                wordcount[w] = 1
            else:
                wordcount[w] += 1

    counts = wordcount.values()
    keys = wordcount.keys()

    sorted_idx = np.argsort(counts)[::-1]

    worddict = dict()

    for idx, ss in enumerate(sorted_idx):
        worddict[keys[ss]] = idx+2  # leave 0 and 1 (UNK)

    print np.sum(counts), ' total words ', len(keys), ' unique words'
    return worddict


def grab_data(reviews, dictionary,scores,score):
    sentences = []

    for idx, sc in enumerate(scores) :
        if sc==score:
            sentences.append(reviews[idx])


    sentences = tokenize(sentences)

    seqs = [None] * len(sentences)
    for idx, ss in enumerate(sentences):
        words = ss.strip().lower().split()
        seqs[idx] = [dictionary[w] if w in dictionary else 1 for w in words]

    return seqs


def main(reviews,scores,test=False):
    # Get the dataset from http://ai.stanford.edu/~amaas/data/sentiment/
    if test:
        f = open('data.dict.pkl', 'rb')

        dictionary = pickle.load(f)
        f.close()

    else:
        dictionary = build_dict(reviews)

    train_x_1 = grab_data(reviews, dictionary,scores,score = 1)
    train_x_2 = grab_data(reviews, dictionary,scores,score = 2)
    train_x_3 = grab_data(reviews, dictionary,scores,score = 3)
    train_x_4 = grab_data(reviews, dictionary,scores,score = 4)
    train_x_5 = grab_data(reviews, dictionary,scores,score = 5)
    train_x = train_x_1 + train_x_2 + train_x_3 + train_x_4 + train_x_5
    # train_y = [[0, 0, 0, 0, 1]] * len(train_x_1) + [[0, 0, 0, 1, 0]] * len(train_x_2) + [[0, 0, 1, 0, 0]] * len(train_x_3) + [[0, 1, 0, 0, 0]] * len(train_x_4) + [[1, 0, 0, 0, 0]] * len(train_x_5)
    train_y = [[1.0]] * len(train_x_1) + [[2.0]] * len(train_x_2) + [[3.0]] * len(train_x_3) + [[4.0]] * len(train_x_4) + [[5.0]] * len(train_x_5)

    if not test:
        if os.path.isfile('data.pkl'):
            os.remove('data.pkl')
        f = open('data.pkl', 'wb')
        pkl.dump((train_x, train_y), f, -1)
        f.close()

        if os.path.isfile('data.dict.pkl'):
            os.remove('data.dict.pkl')
        f = open('data.dict.pkl', 'wb')
        pkl.dump(dictionary, f, -1)
        f.close()
    else:
        if os.path.isfile('data_test.pkl'):
            os.remove('data_test.pkl')
        f = open('data_test.pkl', 'wb')
        pkl.dump((train_x, train_y), f, -1)
        f.close()

    return train_x,train_y,dictionary

if __name__ == '__main__':
    reviews = np.load('review.npy')
    scores = np.load('scores.npy')
    train_x,train_y,dictionary = main(reviews,scores)