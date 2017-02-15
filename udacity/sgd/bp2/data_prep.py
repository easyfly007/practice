import numpy as np
import pandas as pd

admissions = pd.read_csv('binary.csv')
data = pd.concat([admissions, pd.get_dummies(admissions['rank'], prefix = 'rank')], axis = 1)
data.drop('rank', axis = 1)
for field in ['gre', 'gpa']:
	mean, std = data[field].mean(), data[field].std()
	data.loc[:, field] = (data[field]-mean) / std

np.random.seed(21)
sample = np.random.choice(data.index, size = int(len(data)*0.9), replace = False)
train_data, test_data = data.ix[sample], data.drop(sample)

features, targets = train_data.drop('admit', axis = 1), train_data['admit']
features = features.drop('rank', axis = 1)
features_test, targets_test = test_data.drop('admit', axis = 1), test_data['admit']
features_test = features_test.drop('rank', axis = 1)
