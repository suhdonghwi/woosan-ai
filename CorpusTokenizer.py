from konlpy.corpus import CorpusLoader
from konlpy.tag import Twitter


def load_modern():
    modern_loader = CorpusLoader('modern')
    return modern_loader


def make_doc(loader):
    result = []
    for filepath in loader.fileids():
        with loader.open(filepath) as corpus_file:
            result = result + corpus_file.read().splitlines()

    return result


def filter_puntuation(sent):
    return list(filter(lambda ch: ch not in ['.', ',', '=', '-', '~'], sent))


def tokenize_doc(doc):
    twitter = Twitter()
    return list(map(lambda sent: filter_puntuation(twitter.morphs(sent)), doc))


if __name__ == "__main__":
    print(tokenize_doc(make_doc(load_modern())))
