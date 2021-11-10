#!/usr/bin/env python
# coding: utf-8

# In[16]:


from keras.datasets import mnist
from keras.utils.np_utils import to_categorical 
import numpy as np 
import pandas as pd

from sklearn.preprocessing import minmax_scale 
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Model

from keras.models import Sequential 
from keras.layers import Input,Dense, Dropout, Activation, Flatten, concatenate
from tensorflow.keras.optimizers import Adam


# In[17]:


from keras.models import load_model
model = load_model('LCK.h5')


# In[47]:




def showMatchResult(dragons,barons,towers,model):
    x = np.array([dragons])
    y = np.array([barons])
    z = np.array([towers])
    x = np.asarray(x).astype('int32').reshape((-1,1))
    y = np.asarray(y).astype('int32').reshape((-1,1))
    z = np.asarray(z).astype('int32').reshape((-1,1))
    
    preds = model.predict([x,y,z])
    return preds

def showMatchResult2(x,model):
    preds = model.predict([x])
    return preds

#test 코드
preds = showMatchResult([1,-3,-6],[1,4,7],[3,-3,3],model)


# In[48]:


'''x는 [blue팀 드래곤 처치수, blue팀 바론 처치수, blue팀 타워 철거수, red팀 드래곤 처치수, red팀 바론 처치수, red팀 타워 철거수]의 형태로
구성된 배열'''



def predict_value(x):
    x = np.array(x)
    y = model.predict(x)
    return y
 
    


# In[49]:


print(preds)


# In[ ]:




