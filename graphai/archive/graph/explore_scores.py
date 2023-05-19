import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

values_in = pd.Series([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000], name='in')
values_out = pd.Series([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000], name='out')

df = pd.merge(values_in, values_out, how='cross')
df['min_in_out'] = df.min(axis=1)
df['max_in_out'] = df.max(axis=1)

# df['r'] = (1 + df['min_in_out']) / (1 + df['max_in_out'])
df['r'] = df['min_in_out'] / df['max_in_out']

# df['score'] = np.tanh(0.5 * df['r'] * np.log(df['out']))
# df['score'] = (np.power(df['out'], df['r']) - 1) / (np.power(df['out'], df['r']) + 1)
# df['score'] = (np.power(df['out'], df['r']) - 1) / (np.power(df['out'], df['r']) + (100 / (df['out'] + df['in'])))
# df['score'] = (np.power(df['out'], df['r']) - 1) / (np.power(df['out'], df['r']) + 1/np.power(df['max_in_out'], df['r']))
# df['score'] = (np.power(df['out'], df['r']) - 1) / (np.power(df['out'], df['r']) + 100 * 1 / (np.power(df['in'], 1) + np.power(df['out'], 1)))

df['score'] = 1 - 1 / (1 + np.log(1 + df['r'] * np.log(df['out'])))
# df['score'] = df['r'] * df['in']

df = df.pivot(index='in', columns='out', values='score')

sns.heatmap(df, annot=True, vmin=0, vmax=1)
plt.show()
