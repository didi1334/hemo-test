import io

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree

hemo = pd.read_csv('hemog1.csv')
# print(hemo.head())

X = hemo.loc[:, "red":'hemoglobin']
y = hemo['anemia']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

clf = DecisionTreeClassifier(criterion='entropy', max_depth=4)
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print('Accuracy on training set: {:.3f}'.format(clf.score(X_train, y_train)))
print('Accuracy on test set: {:.3f}'.format(clf.score(X_test, y_test)))

from sklearn.tree import export_graphviz
import graphviz
from sklearn.externals import StringIO
from IPython.display import Image
import pydotplus

dot_data = export_graphviz(clf, out_file=None,
                           filled=True, rounded=True,
                           special_characters=True,
                           feature_names=('ftr_red', 'ftr_green', 'ftr_blue', 'ftr_rgb', 'ftr_sex',
                                          'ftr_age', 'ftr_hemoglobin'), class_names=['0', '1'])

graph = graphviz.Source(dot_data)
plt.show()

