import re
import xml.etree.ElementTree as ET
import sys
import os
import nltk


nltk.download('rslp')

DEBUG = False
DISTANCE = "jaccard"


def main():
    stopWords = open("stopwords.txt", "r+").read().splitlines()[1:]

    if not os.path.isfile(sys.argv[1]):
        print("[ERROR] File {} does not exist.".format(sys.argv[1]))
        sys.exit()

    if not os.path.isfile(sys.argv[2]):
        print("[ERROR] File {} does not exist.".format(sys.argv[2]))
        sys.exit()

    listaPerguntasFAQ = extractXML(sys.argv[1])

    for id in listaPerguntasFAQ:
        listaPerguntasFAQ[id] = tokStem(removeStopWords(preProc(listaPerguntasFAQ[id]), stopWords))     #EXP 2 & 3
        # listaPerguntasFAQ[id] = removeStopWords(preProc(listaPerguntasFAQ[id]), stopWords)              #EXP 1

    listaPerguntasTXT = tokStem(removeStopWords(preProc(extractTXT(sys.argv[2])), stopWords))           #EXP 2 & 3
    # listaPerguntasTXT = removeStopWords(preProc(extractTXT(sys.argv[2])), stopWords)                    #EXP 1

    if (DEBUG):
        ids, debug, stats = similarity(listaPerguntasFAQ, listaPerguntasTXT, distance=DISTANCE, debug=True)

        for info in debug:
            print(info + '\n')

        print(stats)

    else:
        ids = similarity(listaPerguntasFAQ, listaPerguntasTXT, distance=DISTANCE)

    open("resultados.txt", "w+").write(str(ids))


def similarity(listaPerguntasFAQ, listaPerguntasTXT, distance="jaccard", debug=False):
    ids = []
    info = []
    tp = fp = tn = fn = corrects = 0
    distance_index = {
        "edit": (edit_distance, 1000),
        "jaccard": (jaccard_distance, 0.7),
        "dice": (dice_distance, 0.6),
    }
    distance_function, min_distance = distance_index[distance]

    for test, line in zip(listaPerguntasTXT, range(1, len(listaPerguntasTXT) + 1)):
        best_sentence = ""
        best = min_distance
        bestId = '0'
        splitted_test = test.split()
        test_id = '0'

        if debug:
            test_id = splitted_test.pop(-1)

        for key in listaPerguntasFAQ:
            for sentence in listaPerguntasFAQ[key]:
                result = distance_function(sentence.split(), splitted_test)

                if result < best:
                    best_sentence = sentence
                    bestId = key
                    best = result

        ids.append(bestId)

        if debug:
            if bestId == test_id:
                t_result = bcolors.accepted("ACCEPTED")
                t_id = bcolors.accepted(bestId)
                corrects += 1
                if test_id == '0':  # Verdadeiramente negativo (t_id = p_id = 0)
                    tn += 1
                else:  # verdadeiramente positivo, com id correto (t_id = p_id > 0)
                    tp += 1
            else:
                t_result = bcolors.failed("FAILED")
                t_id = bcolors.failed(bestId)
                if test_id == '0':  # Falsamente positivo (t_id = 0 , p_id > 0)
                    fp += 1
                elif bestId == '0':  # falsamente negativo (t_id > 0, p_id = 0)
                    fn += 1
                else:  # verdadeiramente positivo, com id incorreto (t_id > 0,  p_id > 0, t_id |= p_id)
                    tp += 1

            info.append("        LINE: " + str(line) +
                        "\n      TESTED: " + test +
                        "\nAPPROXIMATED: " + best_sentence + " " + t_id +
                        "\n    DISTANCE: " + str(best) +
                        "\n      RESULT: " + t_result)

    if debug:
        p_positives = tp + fp
        t_positives = tp + fn
        trues = tp + tn
        total = len(ids)

        stats = ("\n" + bcolors.highlight("STATS") +
                 "\n   TRUE POSITIVES: " + str(tp) +
                 "\n  FALSE POSITIVES: " + str(fp) +
                 "\n   TRUE NEGATIVES: " + str(tn) +
                 "\n  FALSE NEGATIVES: " + str(fn) +
                 "\n        PRECISION: " + str(tp) + "/" + str(p_positives) + " (" + str(tp / p_positives) + ")" +
                 "\n           RECALL: " + str(tp) + "/" + str(t_positives) + " (" + str(tp / t_positives) + ")" +
                 "\n   CLASS ACCURACY: " + str(trues) + "/" + str(total) + " (" + str(trues / total) + ")" +
                 "\n      ID ACCURACY: " + str(corrects) + "/" + str(total) + " (" + str(corrects / total) + ")" +
                 "\n")

        return ids, info, stats

    return ids


def extractXML(filename):
    tree = ET.parse(filename)
    root = tree.getroot()

    documents = {}

    for child in root:
        for j in range(0, len(child[1])):
            sentences = []
            for i in range(0, 4):
                sentences.append(child[1][j][1][i].text)
            documents[child[1][j][2].get('id')] = sentences

    return documents


def extractTXT(filename):
    return open(filename).read().splitlines()


# ------------------------------
# Pre-processing
# ------------------------------

def removeStopWords(list, stopWordList):
    perguntas = []
    for sentence in list:
        sentence = sentence.split()
        frase = []
        for word in sentence:
            if word.lower() not in stopWordList:
                frase.append(word)
            fraseAux = ' '.join(frase)
        perguntas.append(fraseAux)
    return perguntas


def tokStem(perguntas):
    perguntas_tok_stem = []
    stemmer = nltk.stem.RSLPStemmer()
    for l in perguntas:
        l = nltk.word_tokenize(l)
        l1 = []
        for word in l:
            word = stemmer.stem(word)
            l1.append(word)
        l = ' '.join(l1)
        perguntas_tok_stem.append(l)
    return perguntas_tok_stem


def preProc(Lista):
    perguntas = []
    for l in Lista:
        # ELIMINA ACENTOS
        # l = re.sub(u"ã", 'a', l)
        # l = re.sub(u"á", "a", l)
        # l = re.sub(u"à", "a", l)
        # l = re.sub(u"õ", "o", l)
        # l = re.sub(u"ô", "o", l)
        # l = re.sub(u"ó", "o", l)
        # l = re.sub(u"é", "e", l)
        # l = re.sub(u"ê", "e", l)
        # l = re.sub(u"í", "i", l)
        # l = re.sub(u"ú", "u", l)
        # l = re.sub(u"ç", "c", l)
        # l = re.sub(u"Ã", 'A', l)
        # l = re.sub(u"Á", "A", l)
        # l = re.sub(u"À", "A", l)
        # l = re.sub(u"Õ", "O", l)
        # l = re.sub(u"Ô", "O", l)
        # l = re.sub(u"Ô", "O", l)
        # l = re.sub(u"Ó", 'O', l)
        # l = re.sub(u"Í", "I", l)
        # l = re.sub(u"Ú", "U", l)
        # l = re.sub(u"Ç", "C", l)
        # l = re.sub(u"É", "E", l)
        # TUDO EM MINÚSCULAS
        l = l.lower()
        # ELIMINA PONTUAÇÃO
        l = re.sub("[?|\.|!|:|,|;|/]", '', l)
        perguntas.append(str(l))
    return perguntas


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @classmethod
    def failed(cls, text):
        return cls.FAIL + cls.BOLD + text + cls.ENDC

    @classmethod
    def accepted(cls, text):
        return cls.OKGREEN + cls.BOLD + text + cls.ENDC

    @classmethod
    def highlight(cls, text):
        return cls.HEADER + cls.BOLD + text + cls.ENDC


def edit_distance(label1, label2):
    return nltk.edit_distance(label1, label2, 2)


def jaccard_distance(label1, label2):
    return nltk.jaccard_distance(set(label1), set(label2))


def dice_distance(label1, label2):
    """Distance metric comparing set-similarity.
    """
    set1, set2 = set(label1), set(label2)
    return 1 - (2 * len(set1.intersection(set2))) / (len(set1) + len(set2))


main()
