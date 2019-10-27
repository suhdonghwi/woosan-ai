from gensim.models.word2vec import Word2Vec


def train_model(tokenized):
    model = Word2Vec(tokenized, workers=8, iter=5,
                     size=300, window=10, min_count=20)
    model.init_sims(replace=True)
    return model


def load_model(path):
    model = Word2Vec.load(path)
    return model
