import json
from flask import Flask, jsonify, request
from konlpy.tag import Komoran

from utils.sentence import similar_sentences
from train.word_embed import load_model

analyzer = Komoran()
model = load_model('./train/data/word2vec.model')
app = Flask(__name__)


@app.route('/api/similar-sentences', methods=['POST'])
def check_similars():
    data = request.get_json()

    sent = data['input']
    key_sents = data['keys']

    similar_sents = similar_sentences(model, analyzer, sent)
    for sent in similar_sents:
        for (i, tag) in enumerate(sent):
            sent[i] = tag[0]

    analyzed_key_sents = list(
        map(lambda s: analyzer.morphs(s), key_sents))
    print("Similar sents")
    print(similar_sents)
    print("Analyzed key sents")
    print(analyzed_key_sents)
    print("----------------")

    if any([s1 == s2 for s1 in similar_sents for s2 in analyzed_key_sents]):
        return jsonify({'detected': True})
    else:
        return jsonify({'detected': False})
