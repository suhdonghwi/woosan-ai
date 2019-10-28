import pickle

from konlpy.corpus import CorpusLoader
from konlpy.tag import Komoran

from WordEmbed import train_model, load_model
from CorpusTokenizer import make_doc, tokenize_doc

analyzer = Komoran()


def train():
    print('Loading corpus...')
    modern_loader = CorpusLoader('modern')

    print('Tokenizing doc...')
    tokenized = tokenize_doc(make_doc(modern_loader))

    print('Dumping doc...')
    with open('./train/tokenized.pkl', 'wb') as doc_file:
        pickle.dump(tokenized, doc_file)

    with open('./train/tokenized.pkl', 'rb') as doc_file:
        tokenized = pickle.load(doc_file)

    print('Training...')
    model = train_model(tokenized)
    model.save("./train/word2vec.model")

    print('Finished! Saved model.')

    return model


def similar_sentences(model, sent):
    result = []
    tags = analyzer.pos(sent)
    print(tags)

    for (i, (word, pos)) in enumerate(tags):
        # print(pos)
        if pos in ['NNG', 'NP', 'NNP', 'VV', 'VA', 'VX', 'MAG', 'MAJ']:
            most_similar_words = [
                word for (word, possibility) in model.wv.most_similar(word)]
            for word in most_similar_words[:2]:
                if pos[0] == analyzer.pos(word)[0][1][0]:
                    similar_tags = [word if i == j else tag[0]
                                    for (j, tag) in enumerate(tags)]
                    result.append(similar_tags)

    return result


if __name__ == "__main__":
    model = load_model('./train/word2vec.model')
    print(len(model.wv.vocab))

    while True:
        sent = str(input('>> '))

        for similar_sent in similar_sentences(model, sent):
            print(similar_sent)
        # print(model.wv.doesnt_match(word.split()))
        # print(model.wv.most_similar(word))
