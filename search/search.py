from flask import Flask
from flask import request
from flask import render_template
from flask import Markup
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

import os
import gensim

from model_code.execution import jaccard_search, word2vec_search, simple_search
from model_code.model_lib import load_voc, load_articles

app = Flask(__name__)

os.chdir("model_code")

global stop_words
stop_words = stopwords.words('english')
stop_words.extend(["the"])
stop_words = set(stop_words)

global stemmer
stemmer = SnowballStemmer('english')

global articles_list
articles_list = load_voc("filtered_access.csv")

global if_access
if_access = load_voc("if_access.csv")

global tokens_access
tokens_access = load_voc("tokens_access.csv")

global model
model = gensim.models.Word2Vec.load("word2vec.model")


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/', methods=['POST'])
def text_box():
    results = ""
    query = request.form['query']
    all = (request.form['type-product'].lower() == "verbatim")
    topk = int(request.form['topk'])
    method = request.form["method"].lower()
    if method == "simple search":
        results = load_articles(simple_search(query, if_access, stop_words, stemmer, topk, all), articles_list)
    else:
        limit = int(request.form['limit'])
        threshold = float(request.form['threshold'])
        if method == "see also":
            results = load_articles(
                jaccard_search(query, if_access, tokens_access, threshold, stop_words, stemmer, topk, all, limit),
                articles_list)
        else:
            weight = int(request.form['weight'])
            results = load_articles(
                word2vec_search(model, query, stop_words, stemmer, limit, if_access, threshold, topk, all, weight), articles_list)

    return render_template("results.html", message=Markup(results))


if __name__ == '__main__':
    app.run()
