# coding=utf-8

import codecs
import jieba.posseg as pseg
from rank_bm25 import BM25Okapi as BM25

# 构建停用词表
stopwords = codecs.open('preprocess/cn_stopwords.txt','r',encoding='utf8').readlines()
stopwords = [ w.strip() for w in stopwords ]

# jieba里的停用词属性
stop_flag = ['x', 'c', 'u','d', 'p', 't', 'uj', 'm', 'f', 'r']

# 对文字去停用词
def tokenization(text):
    result = []
    words = pseg.cut(text)
    for word, flag in words:
        if flag not in stop_flag and word not in stopwords:
            result.append(word)
    return result

# 文档获取类
class CORPUS_RANK(object):
    def __init__(self, corpus):
        self.tokenized_corpus = [tokenization(doc) for doc in corpus]
        self.bm25 = BM25(self.tokenized_corpus)

    def get_document(self, query):
        tokenized_query = tokenization(query)
        doc_scores = list(self.bm25.get_scores(tokenized_query))
        print(doc_scores)
        if sum(doc_scores)==0:
            return None
        else:
            max_index = doc_scores.index(max(doc_scores))
            return max_index
