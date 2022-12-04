import json
from glob import glob
import re
from nltk import word_tokenize as lib_tokenizer
import string


def preprocess(x, max_length=-1, remove_puncts=False):
    x = nltk_tokenize(x)
    x = x.replace("\n", " ")
    if remove_puncts:
        x = "".join([i for i in x if i not in string.punctuation])
    if max_length > 0:
        x = " ".join(x.split()[:max_length])
    return x


def nltk_tokenize(x):
    return " ".join(word_tokenize(strip_context(x))).strip()


def post_process_answer(x, entity_dict):
    if type(x) is not str:
        return x
    try:
        x = strip_answer_string(x)
    except:
        return "NaN"
    x = "".join([c for c in x if c not in string.punctuation])
    x = " ".join(x.split())
    y = x.lower()
    if len(y) > 1 and y.split()[0].isnumeric() and ("tháng" not in x):
        return y.split()[0]
    if not (x.isnumeric() or "ngày" in x or "tháng" in x or "năm" in x):
        if len(x.split()) <= 2:
            return entity_dict.get(x.lower(), x)
        else:
            return x
    else:
        return y


dict_map = dict({})


def word_tokenize(text):
    global dict_map
    words = text.split()
    words_norm = []
    for w in words:
        if dict_map.get(w, None) is None:
            dict_map[w] = ' '.join(lib_tokenizer(w)).replace('``', '"').replace("''", '"')
        words_norm.append(dict_map[w])
    return words_norm


def strip_answer_string(text):
    text = text.strip()
    while text[-1] in '.,/><;:\'"[]{}+=-_)(*&^!~`':
        if text[0] != '(' and text[-1] == ')' and '(' in text:
            break
        if text[-1] == '"' and text[0] != '"' and text.count('"') > 1:
            break
        text = text[:-1].strip()
    while text[0] in '.,/><;:\'"[]{}+=-_)(*&^!~`':
        if text[0] == '"' and text[-1] != '"' and text.count('"') > 1:
            break
        text = text[1:].strip()
    text = text.strip()
    return text


def strip_context(text):
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def check_number(x):
    x = str(x).lower()
    return (x.isnumeric() or "ngày" in x or "tháng" in x or "năm" in x)
