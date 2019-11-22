from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re

import copy
import math
import operator
import ast

from model_code.model_lib import filter_all, sorted_access, random_access, load_tokens, access_line, get_articles


def ta(list, k=3, all_terms=0):
    tokens = len(list)
    if tokens == 0:
        return None
    if all_terms:
        list = filter_all(list)

    # initialize the parameters
    C = {}
    T = [math.inf] * tokens

    # start of the algorithm
    while (len(C) < k) or not all(score >= sum(T) / tokens for score in C.values()):
        same = True

        for i in range(0, tokens):
            id, score, T[i] = sorted_access(list, i, T[i])

            if id is not None:
                same = False
                C[id] = score

                for j in [x for x in range(0, tokens) if x != i]:
                    score, T[j] = random_access(list, j, id, T[j])
                    C[id] = C[id] + score

                C[id] = C[id] / tokens
                if len(C) > k:
                    C = dict(sorted(C.items(), key=operator.itemgetter(1), reverse=True))
                    C.popitem()

        if same:
            return (C)

    return (C)


def jaccard(article, other, list):
    union = len(list[str(article)].union(list[str(other)]))
    intersect = len([w for w in list[str(article)] if w in list[str(other)]])
    if union == 0:
        return 0
    else:
        return (intersect / union)

def jaccard_search(query, if_access, tokens_access, threshold, stop_words, stemmer, k=3, all=0, limit=0):
    lists1 = load_tokens(query, if_access, "if.csv", stop_words, stemmer)
    lists2 = copy.deepcopy(lists1)

    articles = ta(lists1.copy(), k, all)
    if limit > 0:
        other_articles = get_articles(lists2, articles.keys())
        list_ = {}

        for article in articles.keys():
            list_[str(article)] = set(
                ast.literal_eval(access_line("tokens.csv", tokens_access[str(article)]).split('#')[1][1:-3]))

        for other in other_articles:
            list_[str(other)] = set(
                ast.literal_eval(access_line("tokens.csv", tokens_access[str(other)]).split('#')[1][1:-3]))

        score = 0.0
        results = {}
        for article in articles.keys():
            for other in other_articles:
                score = jaccard(article, other, list_)
                if score >= threshold:
                    results[other] = score
            other_articles = [doc for doc in other_articles if doc not in results.keys()]

        results = dict(sorted(results.items(), key=operator.itemgetter(1), reverse=True))
        for key, value in results.items():
            articles[key] = value
            limit -= 1
            if limit == 0:
                return articles
    return articles


def word2vec_search(model, query, stop_words, stemmer, limit, if_access, threshold, k=3, all=0, weight=1):
    query = re.sub(r'[^\w]', ' ', query)
    tokens = word_tokenize(re.sub(r"\d+", "", query.lower()))
    tokens = [w for w in tokens if w not in stop_words]
    tokens = set([stemmer.stem(w) for w in tokens])
    results = set(model.wv.most_similar(positive=tokens, topn=limit))
    result = set()
    for word, score in results:
        if score >= threshold:
            print(word, ": ", score)
            if word not in tokens:
                result.add(word)
    list1 = []
    for token in tokens:
        if token in if_access.keys():
            d = dict(ast.literal_eval(access_line("if.csv", if_access[token])))
            d.update((x, y*weight) for x, y in d.items())
            list1.append(d)
    list = [dict(ast.literal_eval(access_line("if.csv", if_access[token]))) for token in result if
            token in if_access.keys()]
    list.extend(list1)
    return ta(list, k, all)


def simple_search(query, if_access, stop_words, stemmer, k=3, all=0):
    lists = load_tokens(query, if_access, "if.csv", stop_words, stemmer)
    articles = ta(lists, k, all)
    return articles

