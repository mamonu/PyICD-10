import pandas as pd

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD






df = pd.read_csv('icd10.csv',encoding = 'latin1')


description = df['description']


vectors = TfidfVectorizer().fit_transform(description)


X_reduced = TruncatedSVD(n_components=50, random_state=0).fit_transform(vectors)



X_embedded = TSNE(n_components=2, verbose=2).fit_transform(X_reduced)




plt.scatter(X_embedded[:, 0], X_embedded[:, 1],
         marker="x")


plt.show()