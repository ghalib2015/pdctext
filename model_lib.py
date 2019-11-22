from nltk import word_tokenize

import re
import ast

import mmap


def load_voc(filename):
    with open(filename, "r") as file:
        voc = ast.literal_eval(file.readline())
    return voc


def load_tokens(query, voc, file, stop_words, stemmer):
    query = re.sub(r'[^\w]', ' ', query)
    tokens = word_tokenize(re.sub(r"\d+", "", query.lower()))
    tokens = [w for w in tokens if w not in stop_words]
    tokens = set([stemmer.stem(w) for w in tokens])
    list = [dict(ast.literal_eval(access_line(file, voc[token]))) for token in tokens if token in voc.keys()]
    return list


def sorted_access(list, index, T):
    if len(list[index]) == 0:
        return None, None, T
    else:
        id, score = next(iter(list[index].items()))
        del list[index][id]
    return id, score, score


def random_access(list, index, id, T):
    if len(list[index]) == 0:
        return 0, 0
    elif id not in list[index]:
        return 0, T
    else:
        score = list[index][id]
        if id == next(iter(list[index].keys())):
            T = score
        del list[index][id]
        return score, T


def access_line(file, index):
    with open(file, "r+") as f:
        mm = mmap.mmap(f.fileno(), 0)
        mm.seek(index)
        return mm.readline().decode().replace('\\', '')


def filter_all(list):
    index = 0
    minimum = len(list[0])
    for l in range(1, len(list)):
        if len(list[l]) < minimum:
            minimum = len(list[l])
            index = l
    for i in [x for x in range(0, len(list)) if x != index]:
        delete = [key for key in list[index] if key not in list[i]]
        for key in delete: del list[index][key]
    for i in [x for x in range(0, len(list)) if x != index]:
        delete = [key for key in list[i] if key not in list[index]]
        for key in delete: del list[i][key]
    return list


def load_articles(list, articles_list):
    results = ""
    for id, score in list.items():
        results += "<p><b>" + str(id) + ": " + str(score) + "<b><br />"
        results += str(access_line("filtered.csv", articles_list[str(id)]).split("#")[1]) + "</p>"
    return results


def get_articles(list, articles):
    results = set()
    for i in range(len(list)):
        for doc in list[i].keys():
            if doc not in articles:
                results.add(doc)
    return results
