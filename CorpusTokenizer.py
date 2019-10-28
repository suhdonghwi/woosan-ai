from functools import reduce

from konlpy.tag import Komoran


analyzer = Komoran()


# CorpusLoader를 기반으로 word2vec 모델에 맞게 doc 데이터를 생성해서 반환하는 함수
def make_doc(loader):
    result = []
    for filepath in loader.fileids():
        with loader.open(filepath) as corpus_file:
            result = result + corpus_file.read().splitlines()

        print('Made doc for ' + filepath)

    print('Finished making doc')
    return result


# doc 데이터를 받고 각 문장마다 형태소 분석해서 반환하는 함수
def tokenize_doc(doc):
    result = []
    size = len(doc)
    for (i, sent) in enumerate(doc):
        print(str(i / size * 100))
        result.append(analyzer.morphs(sent))

    return result
