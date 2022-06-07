from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train_df = pd.read_csv('hemog1.csv')


predictors = ['red','green','blue','rgb','sex','age', 'hemoglobin']
outcome = 'anemia'

new_record = train_df.loc[0:0, predictors]
X = train_df.loc[1:, predictors]
y = train_df.loc[1:, outcome]

kNN = KNeighborsClassifier(n_neighbors=20)
kNN.fit(X, y)
kNN.predict(new_record)
print(kNN.predict_proba(new_record))

nbrs = kNN.kneighbors(new_record)
maxDistance = np.max(nbrs[0][0])

fig, ax = plt.subplots(figsize=(10, 8))
sns.scatterplot(x='hemoglobin',y='red', style = 'anemia',
                hue='anemia', data=train_df, alpha=0.3, ax=ax)
sns.scatterplot(x='hemoglobin',y='red', style = 'anemia',
                hue = 'anemia',
                data = pd.concat([train_df.loc[0:0, :], train_df.loc[nbrs[1][0] + 1,:]]),
                ax = ax, legend=False)
ellipse =Ellipsis(xy = new_record.values[0],
                  width = 2 * maxDistance, height = 2 * maxDistance,
                  edgecolor = 'black', fc = 'None', lw = 1)
ax.add_patch(ellipse)
ax.set_xlim(.25, .29)
ax.set_ylim(0, .03)

plt.tight_layout()
plt.show()
