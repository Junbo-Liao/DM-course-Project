from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from numpy import *
from sklearn.linear_model import LogisticRegression


def MailL2Reg(vec,lab):
    classifier = LogisticRegression(penalty='l2')
    classifier.fit(vec, lab)
    res = classifier.predict(vec)
    print("Accuracy: " + str(getAc(res, lab)))

def getAc(res,lab):
    length = shape(res)[0]
    acc = 0.0
    for i in range(length):
        if res[i] == lab[i]:
            acc += 1.0
    return acc/length

newsgroups = fetch_20newsgroups(subset='test')
tfidf_filter_vec1 = TfidfVectorizer(analyzer='word', stop_words='english')
vec = tfidf_filter_vec1.fit_transform(newsgroups.data)
lab = newsgroups.target

newvec1 = SelectKBest(chi2, k=200).fit_transform(vec, lab)
#newvec2 = SelectKBest(mutual_info_classif, k=200).fit_transform(vec, lab)
MailL2Reg(newvec1,lab)
#MailL2Reg(newvec2,lab)

