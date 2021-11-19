#!/usr/bin/env python
# coding: utf-8

# In[7]:


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


# In[8]:


from keras.models import load_model
model = load_model('LCK.h5')


# In[9]:



'''두 팀 사이의 타워 철거수, 바론 처치수, 드래곤 처치수 차이 및
   두 팀이 상대로 만났을 경우 승리할 확률 등을 리턴한다.
   
'''
def getDiff(team1, team2):
    file_directory='../Preprocess/data_after/'
    data=pd.read_csv("{0}match2.csv".format(file_directory))

    data= data.loc[:,['gameid','side','team','result','dragons','barons','towers']]
    condition1 =(data.team == team1)
    condition2 =(data.team == team2)
    
    
    red = data[condition1]
    blue = data[condition2]
    
    red_blue = pd.merge(red, blue, left_on='gameid', right_on='gameid', how='inner')
   

    team1_train = red_blue.loc[:,['result_x','dragons_x','barons_x','towers_x']]
    team2_train = red_blue.loc[:,['result_y','dragons_y','barons_y','towers_y']]
    
    
    
    
    team1_train = team1_train.to_numpy()
    team2_train = team2_train.to_numpy()
    
    team1_win = team1_train[:,0]
    team2_win = team2_train[:,0]
    
    team1_win = np.mean(team1_win,axis=0)
    team2_win = np.mean(team2_win, axis=0)
    
    team1_train = team1_train[:,1:4]
    team2_train = team2_train[:,1:4]
    

    diff = team1_train - team2_train
    
    print(diff)
    
    diff = np.mean(diff,axis=0)
    

    
    
    return team1_win,team2_win,diff


   
    
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
preds = showMatchResult([1,-3,-6],[1,4,7],[3,-3,3],model) #여기에서 input은 다음과 같은 형태로 주어진다.
                                                           #[1,-3,-6]은 
print(preds)


# In[5]:


'''x는 [blue팀 드래곤 처치수, blue팀 바론 처치수, blue팀 타워 철거수, red팀 드래곤 처치수, red팀 바론 처치수, red팀 타워 철거수]의 형태로
구성된 배열'''



def predict_value(x):
    x = np.array(x)
    y = model.predict(x)
    return y
 
    


# In[6]:


team1_win,team2_win,diff = getDiff('NeJon i-mFori','SK Tilicum T1')
print(team1_win)
print(team2_win)
print(diff) #여기서 diff는 team1의 드래곤 처치수, 바론 처치수, 타워 철거수 이 세가지를 
            #team2의 드래곤 처치수, 바론 처치수, 타워 철거수와 뺄셈을 한 것이다.


# In[ ]:





# In[ ]:





# In[ ]:




