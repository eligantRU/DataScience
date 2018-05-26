from collections import defaultdict, Counter, OrderedDict
from operator import itemgetter
import scipy.spatial
import numpy as np
import re


FILE_NAME = "sentences.txt"


def count_lexems(lexems):
    result = defaultdict(int)
    for lexem in lexems.split():
        result[lexem] += 1
    return dict(result)


def get_lexems(sentence):
    return [lexem for lexem in re.split("[^a-z]", sentence) if len(lexem) is not 0]


def get_lexems_in_sentence(sentence):
    lexems = get_lexems(sentence)
    return [lexem for lexem in count_lexems(" ".join(lexems))]


def merge_dicts(dicts):
    result = {}
    for d in dicts:
        result = dict(Counter(result) + Counter(d))
    return result


def get_unique_lexems(sentences):
    stats = []
    for line in sentences:
        lexems = get_lexems(line)
        stats.append(count_lexems(" ".join(lexems)))

    items = merge_dicts(stats).items()
    m = [list(item) for item in items]
    return [key for (key, value) in m]


def get_counted_sentence(sentence):
    result = {}
    lexems_in_sentence = get_lexems_in_sentence(sentence)
    lexems = get_lexems(sentence)
    for unique_lexem in lexems_in_sentence:
        count = lexems.count(unique_lexem)
        if count is not 0:
            result[unique_lexem] = count
    return result


def get_counted_sentences(sentences):
    counted_sentences = []
    unique_lexems = get_unique_lexems(sentences)
    for sentence in sentences:
        counted_sentence = get_counted_sentence(sentence)
        for unique_lexem in unique_lexems:
            if unique_lexem not in counted_sentence:
                counted_sentence[unique_lexem] = 0
        counted_sentences.append(counted_sentence)
    return counted_sentences


def get_matrix(counted_sentences):
    mat = []
    for counted_sentence in counted_sentences:
        mat.append(list(OrderedDict(sorted(counted_sentence.items())).values()))
    return np.array(mat)


def main():
    inp = open(FILE_NAME, "r")
    sentences = [sentence.lower() for sentence in list(inp)]
    counted_sentences = get_counted_sentences(sentences)

    mat = get_matrix(counted_sentences)

    distances = [scipy.spatial.distance.cosine(mat[0], m) for m in mat]
    print(sorted([(i, distance) for i, distance in enumerate(distances)], key=itemgetter(1)))
    inp.close()


if __name__ == "__main__":
    main()
