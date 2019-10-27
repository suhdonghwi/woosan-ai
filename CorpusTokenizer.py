from konlpy.tag import Twitter


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
