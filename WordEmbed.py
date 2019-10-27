from gensim.models.word2vec import Word2Vec


def train_model(tokenized):
    model = Word2Vec(tokenized, workers=4, iter=300,
                     size=200, window=5, min_count=10)
    model.init_sims(replace=True)
    return model


def load_model(path):
    model = Word2Vec.load(path)
    return model
