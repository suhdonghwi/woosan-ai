from gensim.models.word2vec import Word2Vec


# 토큰화된 데이터를 기반으로 word2vec을 학습시키는 함수
def train_model(tokenized):
    model = Word2Vec(tokenized, workers=8, iter=5,
                     size=300, window=10, min_count=20)
    model.init_sims(replace=True)
    return model


# 학습된 모델 데이터를 로드하는 함수
def load_model(path):
    model = Word2Vec.load(path)
    return model
