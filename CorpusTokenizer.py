from functools import reduce

from konlpy.tag import Mecab


analyzer = Mecab()


def make_doc(loader):
    result = []
    for filepath in loader.fileids():
        with loader.open(filepath) as corpus_file:
            result = result + corpus_file.read().splitlines()

        print('Made doc for ' + filepath)

    print('Finished making doc')
    return result


def tokenize_doc(doc):
    result = []
    size = len(doc)
    for (i, sent) in enumerate(doc):
        print(str(i / size * 100))
        result.append(analyzer.morphs(sent))

    return result
