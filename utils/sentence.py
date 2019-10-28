# word2vec 모델과 문장을 받고 유사한 문장을 출력하는 함수
def similar_sentences(model, analyzer, sent):
    result = []
    tags = analyzer.pos(sent)

    for (i, (word, pos)) in enumerate(tags):
        if pos in ['NNG', 'NP', 'NNP', 'VV', 'VA', 'VX', 'MAG', 'MAJ']:
            most_similar_words = [
                word for (word, possibility) in model.wv.most_similar(word)]
            for word in most_similar_words[:2]:
                if pos[0] == analyzer.pos(word)[0][1][0]:
                    similar_tags = [word if i == j else tag[0]
                                    for (j, tag) in enumerate(tags)]
                    result.append(similar_tags)

    return result
