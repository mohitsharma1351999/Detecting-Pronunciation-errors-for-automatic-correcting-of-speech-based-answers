import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

csv = pd.read_csv("train_arsh.csv")
csv.info()
x = csv['shape']
x = x.astype(int)
x.head()
plt.figure()
ax = sns.distplot(x,
                  bins=100,
                  kde=True,
                  color='skyblue',
                  hist_kws={"linewidth": 15, 'alpha': 1})
ax.set(xlabel='width of mel', ylabel='Frequency')
plt.show()
plt.close()
