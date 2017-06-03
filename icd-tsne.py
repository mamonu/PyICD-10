import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import decomposition
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns








df = pd.read_csv('icd10.csv',encoding = 'latin1')


description = df['description']
print description.shape

vectors = CountVectorizer().fit_transform(description)


X_reduced = TruncatedSVD(n_components=50, random_state=0).fit_transform(vectors)

tsne = TSNE(n_components=2, verbose = 2,random_state=0)

Z = tsne.fit_transform(X_reduced)
dftsne = pd.DataFrame(Z, columns=['x','y'])


print dftsne
ax = sns.lmplot('x', 'y', dftsne, fit_reg=False, size=8
                , scatter_kws={'alpha': 0.7, 's': 60})

ax.show()