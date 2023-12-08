import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

values_in = pd.Series([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000], name='in')
values_out = pd.Series([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000], name='out')

df = pd.merge(values_in, values_out, how='cross')
df['min_in_out'] = df.min(axis=1)
df['max_in_out'] = df.max(axis=1)


def yves_score(df, x='out', y='in'):
    r = df['min_in_out'] / df['max_in_out']
    return r * df[y]


def francisco_score(df, x='out', y='in'):
    r = (1 + df['min_in_out']) / (1 + df['max_in_out'])
    return 1 - 1 / (1 + np.log(1 + r * np.log(df[x])))


def aitor_score(df, x='out', y='in', k=50):
    r = (1 + df['min_in_out']) / (1 + df['max_in_out'])
    power = np.power(df[x], r)
    return (power - 1) / (power + (2 * k / (df[x] + df[y])))


def yves_score_sym(df):
    s1 = yves_score(df, x='out', y='in')
    s2 = yves_score(df, x='in', y='out')
    return (s1 + s2) / 2


def francisco_score_sym(df):
    s1 = francisco_score(df, x='out', y='in')
    s2 = francisco_score(df, x='in', y='out')
    return (s1 + s2) / 2


def aitor_score_sym(df, k=50):
    s1 = aitor_score(df, x='out', y='in', k=k)
    s2 = aitor_score(df, x='in', y='out', k=k)
    return (s1 + s2) / 2


df['score'] = aitor_score_sym(df)

df = df.pivot(index='in', columns='out', values='score')

sns.heatmap(df, annot=True, vmin=0, vmax=1)
plt.show()
