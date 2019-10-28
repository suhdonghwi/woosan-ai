import pickle

from konlpy.corpus import CorpusLoader
from konlpy.tag import Komoran

from train.word_embed import train_model, load_model
from corpus.tokenize import make_doc, tokenize_doc
from utils.sentence import similar_sentences


def train():
    print('Loading corpus...')
    modern_loader = CorpusLoader('modern')

    print('Tokenizing doc...')
    tokenized = tokenize_doc(make_doc(modern_loader))

    print('Dumping doc...')
    with open('./train/data/tokenized.pkl', 'wb') as doc_file:
        pickle.dump(tokenized, doc_file)

    with open('./train/data/tokenized.pkl', 'rb') as doc_file:
        tokenized = pickle.load(doc_file)

    print('Training...')
    model = train_model(tokenized)
    model.save("./train/data/word2vec.model")

    print('Finished! Saved model.')

    return model


if __name__ == "__main__":
    model = load_model('./train/data/word2vec.model')
    analyzer = Komoran()
    print(len(model.wv.vocab))

    while True:
        sent = str(input('>> '))

        for similar_sent in similar_sentences(model, analyzer, sent):
            print(similar_sent)
