import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

manual_scores = pd.Series(np.linspace(0, 1, 51), name='x')
nlp_scores = pd.Series(np.linspace(0, 1, 51), name='y')

df = pd.merge(manual_scores, nlp_scores, how='cross')

# df['b'] = pd.concat([df['x'], 0.9 * df['x'] + 0.1 * df['y']], axis=1).max(axis=1)
df['b'] = df['x']

# df['score'] = np.power(df['b'], 1 - df['y'] / 2)
df['score'] = df['b'] * (1 + (2 * df['y'] - 1) / 10)

# df['score'] = df['score'].mask((df['score'] < 0.1) & (df['x'] == 0), 0)
df['score'] = df['score'].clip(0, 1)

df = df.pivot(index='x', columns='y', values='score')

sns.heatmap(df, annot=True)
plt.show()
