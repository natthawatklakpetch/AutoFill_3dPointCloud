import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')

df = pd.read_json('yourdata', orient='records')

dataset = df[:1500]

labels = dataset[['b','g','r']]
b_labels = labels[['b']]
g_labels = labels[['g']]
r_labels = labels[['r']]

dataset = dataset.drop(['b','g','r'], axis=1)

###################################Apply iterate in next pull################################################

x_train,x_test,y_train_b,y_test_b = train_test_split(dataset, b_labels, test_size=0.2, random_state=1)
clf = LinearRegression()
clf.fit(x_train,y_train_b)

b_predictions = clf.predict(x_test)
x_step = np.arange(len(y_test_b))
ax1.plot(x_step,y_test_b,c='blue')
ax1.plot(x_step,b_predictions,c='black')
ax4.plot(x_step,y_test_b,c='blue')

######################################

x_train,x_test,y_train_g,y_test_g = train_test_split(dataset, g_labels, test_size=0.2, random_state=1)
clf = LinearRegression()
clf.fit(x_train,y_train_g)

g_predictions = clf.predict(x_test)
x_step = np.arange(len(y_test_g))
ax2.plot(x_step,y_test_g,c='g')
ax2.plot(x_step,g_predictions,c='black')
ax4.plot(x_step,y_test_g,c='g')

######################################

x_train,x_test,y_train_r,y_test_r = train_test_split(dataset, r_labels, test_size=0.2, random_state=1)
clf = LinearRegression()
clf.fit(x_train,y_train_r)

r_predictions = clf.predict(x_test)
x_step = np.arange(len(y_test_r))
ax3.plot(x_step,y_test_r,c='r')
ax3.plot(x_step,r_predictions,c='black')
ax4.plot(x_step,y_test_r,c='g')

######################################

# ##plot original
#
# d = {'x':x_train['x'],'y':x_train['y'],'z':x_train['z'],'r':y_train_r['r'],'g':y_train_g['g'],'b':y_train_b['b']}
# ddf = pd.DataFrame(data=d)
# train_dataset = ddf.reset_index()
# train_dataset.drop(['index'], axis=1,inplace=True)
# print(train_dataset)
#
# for row in train_dataset.itertuples():
#
#     cl = [row[4], row[5], row[6]]
#     ca = np.asarray(cl)
#
#     ax.scatter(row[1], row[2], row[3], c=ca/255.0)
#
# ax.set_xlabel('x axis')
# ax.set_ylabel('y axis')
# ax.set_zlabel('z axis')
#
# plt.show()

##plot test
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

d = {'x':x_test['x'],'y':x_test['y'],'z':x_test['z'],'r':y_test_r['r'],'g':y_test_g['g'],'b':y_test_b['b']}
ddf = pd.DataFrame(data=d)
test_dataset = ddf.reset_index()
test_dataset.drop(['index'], axis=1,inplace=True)
print(test_dataset)

for row in test_dataset.itertuples():

    cl = [row[4], row[5], row[6]]
    ca = np.asarray(cl)

    ax.scatter(row[1], row[2], row[3], c=ca / 255.0)

ax.set_xlabel('x axis')
ax.set_ylabel('y axis')
ax.set_zlabel('z axis')

##plot prediction
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def lemon(data):
    lime=np.asarray(data)
    lime=np.squeeze(lime)
    return lime


d = {'x':x_test['x'],'y':x_test['y'],'z':x_test['z'],'r':lemon(r_predictions),'g':lemon(g_predictions),'b':lemon(b_predictions)}
ddf = pd.DataFrame(data=d)

test_dataset = ddf.reset_index()
test_dataset.drop(['index'], axis=1,inplace=True)

for row in test_dataset.itertuples():

    cl = [row[4], row[5], row[6]]
    ca = np.asarray(cl)

    ax.scatter(row[1], row[2], row[3], c=ca / 255.0)

ax.set_xlabel('x axis')
ax.set_ylabel('y axis')
ax.set_zlabel('z axis')
print('test new branch')
plt.show()
