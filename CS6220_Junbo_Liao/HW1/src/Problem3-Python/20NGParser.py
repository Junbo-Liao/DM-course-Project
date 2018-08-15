import numpy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import euclidean_distances

news = fetch_20newsgroups(subset='all')
tfidf_filter_vec=TfidfVectorizer(analyzer='word',stop_words='english')
X_tfidf = tfidf_filter_vec.fit_transform(news.data)
n_row = X_tfidf.shape[0]
print(n_row)
f = open("20NG_dotProduct.txt",'wb')
for i in range(0,n_row):
    d = numpy.dot(X_tfidf[i], X_tfidf.T)
    arrDense=d.todense()
    numpy.savetxt(f,arrDense,fmt='%.4f')
    print(i)
f.close
f = open("20NG_euclidiandist.txt",'wb')
for i in range(0,n_row):
    d = euclidean_distances(X_tfidf[i], X_tfidf)
    numpy.savetxt(f,d,fmt='%.4f')
    print(i)
f.close()