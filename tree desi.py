import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pydotplus


df = pd.read_csv('hemog1.csv')
print(df.head())
sns.countplot(df['sex'], hue=df['anemia'])
plt.show()
X = df.drop('anemia', axis=1)
y = df[['anemia']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf_model = DecisionTreeClassifier(criterion="gini", random_state=42, max_depth=3, min_samples_leaf=5)
clf_model.fit(X_train, y_train)
y_predict = clf_model.predict(X_test)
print(accuracy_score(y_test, y_predict))
target = list(df['anemia'].unique())
feature_names = list(X.columns)
from sklearn import tree
import graphviz
from sklearn.tree import export_text
r = export_text(clf_model, feature_names=feature_names)
print(r)
dot_data = tree.export_graphviz(clf_model,
                                )
graph = graphviz.Source(dot_data)

pydot_graph = pydotplus.graph_from_dot_data(dot_data)
from IPython.display import Image
Image(pydot_graph.create_png())
graph.save('graph1.jpg')
