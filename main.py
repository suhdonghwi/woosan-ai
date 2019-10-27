import pickle

from konlpy.corpus import CorpusLoader

from WordEmbed import train_model, load_model
from CorpusTokenizer import make_doc, tokenize_doc

if __name__ == "__main__":
    print('Loading corpus...')
    modern_loader = CorpusLoader('modern')

    print('Tokenizing doc...')
    tokenized = tokenize_doc(make_doc(modern_loader))

    print('Dumping doc...')
    with open('./train/tokenized.pkl', 'wb') as doc_file:
        pickle.dump(tokenized, doc_file)

    # with open('./train/tokenized.pkl', 'rb') as doc_file:
    #     tokenized = pickle.load(doc_file)

    print('Training...')
    model = train_model(tokenized)
    model.save("./train/word2vec.model")

    print('Finished! Saved model.')

    # model = load_model("./train/word2vec.model")
    # print(len(model.wv.vocab))

    while True:
        word = str(input('>> '))
        print(model.wv.most_similar(word))
