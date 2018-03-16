# import modules here
import math
import pandas as pd



def multinomial_nb(training_data, sms):
    _, prior, condprob = train_multinomial_nb(training_data)
    if isinstance(sms, list):
        test_tokens = sms
    else:
        test_tokens = tokenize(sms)
    return apply_multinomial_nb(prior, condprob, test_tokens)

def get_training_data(path=None):
    if path is None:
        path = './data.txt'

    raw_data = pd.read_csv(path, sep='\t')
    raw_data.head()

    training_data = []
    for index in range(len(raw_data)):
        training_data.append((get_freq_of_tokens(raw_data.iloc[index].text), raw_data.iloc[index].category))
    return training_data


def tokenize(sms):
    return sms.split(' ')


def get_freq_of_tokens(sms):
    tokens = {}
    for token in tokenize(sms):
        if token not in tokens:
            tokens[token] = 1
        else:
            tokens[token] += 1
    return tokens


def counter_sum(dicts):
    """
    >>> counter_sum([{'a': 1, 'b': 2}, {'a': 1, 'c': 3}]) 
    {'a': 2, 'b': 2, 'c': 3})  
    :param dicts: 
    :return: 
    """
    ret_dict = dict()
    for dic in dicts:
        for key, val in dic.items():
            if key in ret_dict:
                ret_dict[key] += val
            else:
                ret_dict[key] = val
    return ret_dict


def train_multinomial_nb(training_data):
    klasses = ['spam', 'ham']
    vocabulary = set()
    for r in training_data:
        vocabulary.update(set(r[0]))

    prior = dict()
    condprob = dict()

    for klass in klasses:
        klass_data = [r for r in training_data if r[1] == klass]
        prior[klass] = 1. * len(klass_data) / len(training_data)
        tokens_counter = counter_sum([r[0] for r in klass_data])
        for key in list(tokens_counter):
            tokens_counter[key] += 1
        klass_total = 1. * sum(tokens_counter.values())
        for tok in vocabulary:
            condprob[(klass, tok)] = tokens_counter.get(tok, 1) / klass_total
    return vocabulary, prior, condprob


def apply_multinomial_nb(prior, condprob, test_tokens):
    log_base = 10.
    scores = dict()
    for klass in prior:
        score = math.log(prior[klass], log_base)
        for tok in test_tokens:
            if (klass, tok) in condprob:
                score += math.log(condprob[klass, tok], log_base)
        scores[klass] = score
        # print(score, klass)
    return log_base ** (scores['spam'] - scores['ham'])
