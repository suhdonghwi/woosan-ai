import pickle

from konlpy.tag import Komoran

from train.word_embed import load_model
from utils.sentence import similar_sentences



if __name__ == "__main__":
    model = load_model('./train/data/word2vec.model')
    analyzer = Komoran()
    print(len(model.wv.vocab))

    while True:
        sent = str(input('>> '))

        for similar_sent in similar_sentences(model, analyzer, sent):
            print(similar_sent)
