import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_rows', 400)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

df = pd.read_csv('data/5-df-yves.csv')

n = 11
binarise = True

dependent_variable = 'ManualScore'
independent_variables = df.drop(columns=dependent_variable).columns

if binarise:
    df['ManualScore'] = np.sign(df['ManualScore'])

# Create grid of values
coeffs = pd.DataFrame({'Dummy': [0]})
for column in independent_variables:
    coeffs = pd.merge(coeffs, pd.Series(np.linspace(0, 1, n), name=column), how='cross')
    coeffs = coeffs[coeffs.sum(axis=1) <= 1].reset_index(drop=True)

coeffs = coeffs.drop(columns='Dummy')
coeffs = coeffs[coeffs.sum(axis=1) == 1].reset_index(drop=True)

# Compute errors as coeffs · X^T - y
errors = coeffs @ df[independent_variables].transpose() - df[dependent_variable]

# Square errors
sqerrors = errors ** 2

# Sum errors over all observations
sumsqerrors = sqerrors.sum(axis=1)

# Add columns to coefficients with the sum of squared errors
coeffs['SumSqError'] = sumsqerrors

# Sort by sum of squared errors and print top combinations
print('No restrictions')
print(coeffs.sort_values(by='SumSqError').head(10))
for i in range(2, 9):
    print(f'At least {i} nonzero weights')
    print(coeffs[(coeffs != 0).astype(int).sum(axis=1) > i].sort_values(by='SumSqError').head(10))

sns.heatmap(df.corr(), annot=True, vmin=0, vmax=1, cmap='Blues')
plt.show()
