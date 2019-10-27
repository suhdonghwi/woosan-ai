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
    tags = analyzer.pos(sent)

    for (i, (word, pos)) in enumerate(tags):
        print(pos)
        if pos in ['NNG', 'VV', 'VA', 'VX', 'MAG', 'MAJ']:
            similars = model.wv.most_similar(word)

            similar_tags = [similars[0][0] if i == j else tag[0] for (j, tag) in enumerate(tags)]
            print(similar_tags)
    

if __name__ == "__main__":
    # model = train()
    model = load_model('./train/word2vec.model')
    print(len(model.wv.vocab))

    while True:
        word = str(input('>> '))
        similar_sentences(model, word)
        #print(model.wv.most_similar(word))
