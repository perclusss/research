import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
import pickle
import os
import sys

.
.
.
.
.
.

def get_token_context_label(training_set):
    token_context=[]
    label=[]
    for sentence in training_set:
        for token_index in range(0,len(sentence)):
            if token_index==0:
                token_context.append({'token':sentence[token_index][0], 'pos_itself':sentence[token_index][1], \
                                      'pos_next':sentence[token_index+1][1]})
                label.append(sentence[token_index][2])
            elif token_index==len(sentence)-1:
                token_context.append({'token':sentence[token_index][0], 'pos_itself':sentence[token_index][1], \
                                      'pos_next':'*'})
                label.append(sentence[token_index][2])
            else:
                token_context.append({'token':sentence[token_index][0], 'pos_itself':sentence[token_index][1], \
                                      'pos_next':sentence[token_index+1][1]})
                label.append(sentence[token_index][2])
    return token_context,label

if __name__ == '__main__':
    if (3 != len(sys.argv)):
        sys.stderr.write('Usage:' + '\n')
        sys.stderr.write(sys.argv[0] + ' training_data.dat  classifier.dat' + '\n')
        exit()
    if any((os.path.splitext(os.path.basename(file))[1]!='.dat') for file in sys.argv[1:3]):
        sys.stderr.write('Please check data and classifier extension, it must be .dat, then run the program again.' + '\n')
        exit()
    with open(sys.argv[1],'rb') as f:
        training_set=pickle.load(f)
    training_set=np.array(training_set)
    token_context, label=get_token_context_label(training_set)
    encoder=LabelEncoder()
    y=encoder.fit_transform(label)
    vectorizer = DictVectorizer(dtype=float, sparse=True)
    X = vectorizer.fit_transform(token_context)

    clf=LogisticRegression(penalty='l1',C=4.0)
    clf.fit(X,y)
    with open(sys.argv[2], 'wb') as f:
    	pickle.dump([encoder, vectorizer, clf], f)
