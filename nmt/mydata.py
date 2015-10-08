#-*- coding:utf-8 -*-
from theano import tensor as T
import theano
import cPickle
from nltk.probability import FreqDist
from collections import defaultdict
import os,sys,re
import nltk
from theano.tensor.sort import sort,argsort
from theano.ifelse import ifelse
import time
import numpy as np
import theanets as tn
import sklearn
import downhill
import math

DEBUG = True
EOS = "<EOS>" 
UNK = "<UNK>"  

class Mydata(object):
    def __init__(self,batch_size, fname, f, e):
        self.batch_size = batch_size
        self._prepare_(fname, f, e)
        self.current = 0

    def start(self):
        self.current = 0
        pass
    
    def _prepare_(self, fname, f, e):
        ds = __loaddata__(fname, f , e)
        ds_src,ds_dst = zip(*ds)
        # vocab
        self.v_src_wi,v_src_iw = vocab(ds_src, 15, True)
        self.v_dst_wi,v_dst_iw = vocab(ds_dst, 8)
        print "src vocab size:",len(v_src_iw)
        print "dst vocab size:",len(v_dst_iw)
        with open("v_src_wi.pkl","wb") as sp, open("v_dst_wi.pkl","wb") as dp:
            cPickle.dump(self.v_src_wi, sp)
            cPickle.dump(self.v_dst_wi, dp)
        # prepare data
        self.src = prepare_data(ds_src,self.v_src_wi, self.batch_size) # (batch_cnt,2,T,batch_size)
        self.dst = prepare_data(ds_dst,self.v_dst_wi, self.batch_size)
        #test
        s = self.src[0][0][:,0]
        t = self.dst[0][0][:,0]
        print s
        print t
        print index2words(v_src_iw,s)
        print index2words(v_dst_iw,t)
        print " ".join(ds_src[0])
        print " ".join(ds_dst[0])
        del ds
        del ds_src
        del ds_dst

    def __iter__(self):
        return self

    def next(self):# one mini_batch of size self.batch_size
        if self.current >= len(self.src):
            raise StopIteration
        else:
            s,sw = self.src[self.current]
            t,tw = self.dst[self.current]
            self.current += 1
            print len(s),len(t)
            return s,sw,t,tw

def load_data(batch_size):
    fname = "../../dnn/data/seg"
    f = "en"
    e = "cn"
    train = Mydata(batch_size, fname, f , e)
    valid = None
    test = None
    return train, valid, test

def __loaddata__(fname, f, e, reverseSrc=True):
    '''
        fname:input file name prefix
        f,e: suffix of source and target language
        reverseSrc: token sequence of source language will be reversed
        return: list((list(str),list(str)))
    '''
    data_set = []
    with open(fname+'.'+f) as fp1,\
         open(fname+'.'+e) as fp2:
        for src,dst in zip(fp1,fp2):
            a = re.split(r'\s',src)
            if reverseSrc:
                a.reverse()
            data_set.append((a , re.split(r'\s',dst)))
    return data_set

EOS = "<EOS>"
UNK = "<UNK>"
def vocab(data_sets,max_len,has_unk = False):
    '''
        data_sets: list((list(str),list(str)))
        max_len: max length of token allowed in the vocabulary
        two vocabulary dicts: word->index,index->word
    '''
    fd = FreqDist()
    for row in data_sets:
        for tok in row: #src
            fd[tok.decode('utf8')] += 1
    hapaxes = [k for k in fd if fd[k] <= 2]
    print "len(hapaxes):",len(hapaxes)
    for h in hapaxes:
        del fd[h]
    len_key = [(len(k),k) for k in fd]
    print "len_key"
    for k in sorted(len_key,key=lambda x:x[0])[-10:]:
        print k[1]
    c = 0
    for l,k in len_key:
        if l >= max_len:
            c += 1
            del fd[k]
    print "too long words:",c
    vocab_wi = {}
    vocab_iw = {}

    vocab_wi[EOS] = 0
    vocab_iw[0] = EOS
    offset = 1
    if has_unk:
        vocab_wi[UNK] = 1
        vocab_iw[1] = UNK
        offset += 1
    for i,k in enumerate(fd.keys()):
        vocab_iw[i + offset] = k
        vocab_wi[k] = i + offset
    return vocab_wi,vocab_iw

def row2vec(vocab,row):
    vec = []
    for r in row:
        r = r.decode('utf8')
        if r in vocab:
            vec.append(vocab[r])
        elif UNK in vocab:
            vec.append(vocab[UNK])
    vec.append(vocab[EOS])
    return vec

def index2words(vocab,ind):
    return " ".join([vocab[i].decode("utf8") for i in ind])
    
def prepare_data(data_set,vocab_, batch_size):
    train_set = []
    batch_cnt = int(math.ceil(len(data_set)*1.0/batch_size))
    for i in range(batch_cnt):
        mini_batch = [row for row in data_set[i*batch_size:i*batch_size+batch_size]]
        max_len = max(map(len,mini_batch)) + 1 # one <EOS> after each of src and dst
        weights = []
        for j in range(len(mini_batch)):
            assert i*batch_size + j < len(data_set), "error! should never happen!"
            d = data_set[i*batch_size + j ]
            v = row2vec(vocab_,d) #src
            diff = max_len - len(v)
            mini_batch[j] = v + [0.]*diff
            w = [1]*len(v) + [0]*diff
            weights.append(w)
        #print np.array(mini_batch).shape
        #print max_len
        #print mini_batch[0]
        #print weights[0]
        train_set.append((np.array(mini_batch,dtype="int64").T,np.array(weights,dtype="float32").T)) #(T,B)
    return train_set

if __name__ == '__main__':
    ds,_,_ = load_data(10)
    cnt = 0
    for x in ds:
        if cnt <= 10:
            #print x,len(list(x))
            #print x,xm,y,ym
            cnt += 1
        else:
            break
        break



