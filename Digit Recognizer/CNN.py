from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

dataset = pd.read_csv('input/train.csv')
target = dataset[[0]].values.ravel()
target = dataset[[0]].values.ravel()
train = dataset.iloc[:,1:].values
test = pd.read_csv("input/test.csv").values
'''
rf = RandomForestClassifier(n_estimators=100,n_jobs=2)
rf.fit(train, target)
pred = rf.predict(test)

np.savetxt('submission_rand_forest.csv', np.c_[range(1,len(test)+1),pred], \
           delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')
'''

target = target.astype(np.uint8)
train = np.array(train).reshape((-1, 1, 28, 28)).astype(np.uint8)
test = np.array(test).reshape((-1, 1, 28, 28)).astype(np.uint8)
import matplotlib.pyplot as plt
import matplotlib.cm as cm

plt.imshow(train[1729][0]) # draw the picture