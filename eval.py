import editdistance
import numpy as np
import hgtk
#from mecab import Tagger
#tagger = Tagger()
import mecab
mecab = mecab.MeCab()

from trans import sentranslit as trans
from han2one_rev import uniquealp

from random import shuffle

def read_data(filename):
    with open(filename, 'r') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
    return data

def decompose(s):
    dec = hgtk.text.decompose(s)
    res = ''
    for i in range(len(dec)):
        if dec[i] in uniquealp:
            res += dec[i]
    return res

def eval_diff(s1,s2):
    print(s1+'\t'+s2)
    x = editdistance.eval(s1,'')
    y = editdistance.eval('',s2)
    z = editdistance.eval(s1,s2)
    return np.sqrt(z*z/(x*y))

def count_eng(source,target):
    print('Source: ', source, '\n')
    print('Target: ', target, '\n')
    s = mecab.pos(source)
    ss = [z[0] for z in s]
    sss = [hgtk.checker.is_latin1(z) for z in ss]
    t = mecab.pos(target)
    tt = [z[0] for z in t]
    ttt = [hgtk.checker.is_latin1(z) for z in tt]
    if sum(sss) == 0:
        return 1
    else:
        recall = 1 - sum(ttt)/sum(sss)
        return 2*(recall)/(recall+1)

def test_trans_corpus(filename,number):
    corpus = read_data(filename)
    shuffle_len = len(corpus)
    x_len = list(range(shuffle_len))
    shuffle(x_len)
    x_len = x_len[:number]
    print('\n TEST will be done for randomly selected 10 cases:\n')
    corpus = [corpus[int(z)] for z in x_len]
    result = [eval_diff(decompose(trans(str(z).split(' ')[0][2:])),decompose(str(z).split(' ')[1][:-2])) for z in corpus]
    acc = sum(result)/len(corpus)
    print('\n TEST Transliteration performance: ', acc, '\n')

def test_eng_corpus(filename,number):
    corpus = read_data(filename)
    shuffle_len = len(corpus)
    x_len = list(range(shuffle_len))
    shuffle(x_len)
    x_len = x_len[:number]
    print('\n TEST will be done for randomly selected 10 cases:\n')
    corpus = [corpus[int(z)] for z in x_len]
    result = [count_eng(str(z)[2:-2],trans(str(z)[2:-2])) for z in corpus]
    acc = sum(result)/len(corpus)
    print('\n TEST English word detection performance: ', acc, '\n')

#test_eng_corpus('test/trans_test.txt')
#test_eng_corpus('test/eng_test.txt')



