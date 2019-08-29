
# coding: utf-8

# # Importing Libraries 

# In[12]:


from __future__ import division
import numpy as np
import matplotlib.pyplot as plt 

from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
np.random.seed(42)


# In[13]:


print(load_wine().DESCR)


# In[14]:


wine_data = load_wine()
X = wine_data.data
Y = wine_data.target.reshape((-1,1))


# In[15]:


y = np.zeros((X.shape[0],3))
for i in range(len(Y)):
    y[i][Y[i]] = 1


# In[16]:


X = X.T
for i in range(len(X)):
    X[i] = X[i]/np.max(X[i])
X = X.T
X.shape


# In[17]:


(X,y) = shuffle(X,y,random_state = 40)


# In[18]:


X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.20)


# In[19]:


def relu(x):
    x[x<0]=0
    return x

def softmax(x):
    return np.exp(x)/(np.sum(np.exp(x),axis=0))

def diff_relu(x):
    y = np.zeros(x.shape)
    y[x<=0] = 0
    y[x>0] = 1
    return y

def weight_init(x,y):
    return np.sqrt(2.0/(x+y))*np.random.normal(0,1,(x,y))


# In[20]:


inp_dim = 13
hl1_units = 15
hl2_units = 15
out_dim = 3

W1 = weight_init(hl1_units,inp_dim)
b1 = weight_init(hl1_units,1)
W2 = weight_init(hl2_units,hl1_units)
b2 = weight_init(hl2_units,1)
W3 = weight_init(out_dim,hl2_units)
b3 = weight_init(out_dim,1)


# # Fitting training data using Adagrad approach for adaptive learning rate

# In[21]:


epochs = 1000
epsilon = 0.001
delta = 1e-7

for i in range(epochs):
    (x_train,y_train) = shuffle(X_train,Y_train,random_state = 40)
    loss = 0
    r = {"W1":np.zeros(W1.shape), "W2": np.zeros(W2.shape), "W3": np.zeros(W3.shape), "b1":np.zeros(b1.shape), "b2":np.zeros(b2.shape), "b3":np.zeros(b3.shape)}
    for j in range(0,len(x_train)):
        
        a1 = relu(np.matmul(W1,x_train[j]).reshape((-1,1)) + b1)
        a2 = relu(np.matmul(W2,a1).reshape((-1,1)) + b2)
        x = np.matmul(W3,a2) + b3
        a3 = softmax(x)
        loss += -np.log(a3[np.argmax(y_train[j])]) 
        
        delta3 = a3 - y_train[j].reshape(-1,1)
        delta2 = np.matmul(W3.T,delta3)*diff_relu(a2)
        delta1 = np.matmul(W2.T,delta2)*diff_relu(a1)
        
        grd_b3 = delta3
        grd_W3 = np.matmul(delta3,a2.T)
        grd_b2 = delta2
        grd_W2 = np.matmul(delta2,a1.T)
        grd_b1 = delta1
        grd_W1 = np.matmul(delta1,x_train[j].reshape((-1,1)).T)
        
        r["W1"] += grd_W1*grd_W1
        r["b1"] += grd_b1*grd_b1
        r["W2"] += grd_W2*grd_W2
        r["b2"] += grd_b2*grd_b2
        r["W3"] += grd_W3*grd_W3
        r["b3"] += grd_b3*grd_b3
        
        W1 += -epsilon*grd_W1/(delta+np.sqrt(r["W1"]))
        b1 += -epsilon*grd_b1/(delta+np.sqrt(r["b1"]))
        W2 += -epsilon*grd_W2/(delta+np.sqrt(r["W2"]))
        b2 += -epsilon*grd_b2/(delta+np.sqrt(r["b2"]))
        W3 += -epsilon*grd_W3/(delta+np.sqrt(r["W3"]))
        b3 += -epsilon*grd_b3/(delta+np.sqrt(r["b3"]))
        
    if i%10 == 0:
        print (str(i) + ":" + " loss =" + str(loss/len(x_train)))


# # Model Accuracy

# In[22]:


y_predicted = list()
y_actual = list()
for j in range(0,len(X_test)):
    a1 = relu(np.matmul(W1,X_test[j]).reshape((-1,1)) + b1)
    a2 = relu(np.matmul(W2,a1).reshape((-1,1)) + b2)
    x = np.matmul(W3,a2) + b3
    out = softmax(x)
    y_predicted.append(np.argmax(out))
    y_actual.append(np.argwhere(Y_test[j] == 1)[0][0])
print ("actual out:"+str(y_actual))
print ("predic out:"+str(y_predicted))

