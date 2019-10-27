import re
from os import listdir
from os.path import isfile, join


def extract_script(content):
    result = re.findall(r'<p>(.+)<[/]p>', content)
    result = list(map(lambda line: re.sub(
        r'^-|「|」|[(][^)]*[)]|<[^>]*>', '', line).strip(), result))
    result = list(filter(lambda line: line != '', result))
    return '\n'.join(result)


def refine_corpus_file(source, dest):
    with open(source, 'rt', encoding='UTF16') as raw_corpus:
        content = raw_corpus.read()

    refined_content = extract_script(content).encode('UTF8').decode('UTF8')

    with open(dest, 'w', encoding='UTF8') as refined_corpus:
        refined_corpus.write(refined_content)


if __name__ == "__main__":
    corpus_path = ".\\corpus"
    corpus_files = [f for f in listdir(
        corpus_path) if isfile(join(corpus_path, f))]
    for source_corpus in corpus_files:
        refine_corpus_file(join(corpus_path, source_corpus),
                           join(corpus_path, 'refined', source_corpus))
