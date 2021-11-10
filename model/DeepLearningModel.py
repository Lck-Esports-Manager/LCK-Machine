#!/usr/bin/env python
# coding: utf-8

# In[122]:


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


# In[123]:


file_directory='../Preprocess/data_after/'
#opposite과 my_team에 대한 변수를 두어 그 값이 1이면 내 팀, 0이면 다른 팀 인 방식으로 구현
#테스트 데이터는 match.csv의 result attribute를 사용
#train_data에 사용되는 데이터는 player1.csv의 선수의 티어 값을 이용


# In[124]:


#match.csv 파일을 읽어서 팀의 승패 결과를 test 데이터로 활용
data=pd.read_csv("{0}match2.csv".format(file_directory))



data= data.loc[:,['gameid','side','result','dragons','barons','towers']]


X_train = data[['dragons','barons','towers']]
y_train = data['result']


#X_train 데이터와 y_train 데이터를 numpy 배열로 변환
X_train = X_train.to_numpy()
y_train = y_train.to_numpy()

print(X_train)

dragons_train = X_train[:,0]
print(dragons_train.shape)

print(dragons_train)

#train 데이터의 0번열을 dragons_train에 저장
dragons_train_blue = X_train[0::2,0]
#train 데이터의 1번 열을 barons_train에 저장
barons_train_blue = X_train[0::2,1]
#towers 데이터의 2번 열을 towers_train에 저장
towers_train_blue = X_train[0::2,2]


#num = dragons_train_blue.shape[0]-1
#dragons_train_blue = dragons_train_blue[0:num]
#barons_train_blue = dragons_train_blue[0:num]
#towers_train_blue = dragons_train_blue[0:num]


#train 데이터의 0번열을 dragons_train에 저장
dragons_train_red = X_train[1::2,0]
#train 데이터의 1번 열을 barons_train에 저장
barons_train_red = X_train[1::2,1]
#towers 데이터의 2번 열을 towers_train에 저장
towers_train_red = X_train[1::2,2]

dragons_train = np.zeros_like(dragons_train)
barons_train = np.zeros_like(dragons_train)
towers_train= np.zeros_like(dragons_train)


dragons_train[0::2]= dragons_train_blue - dragons_train_red
barons_train[0::2] = barons_train_blue - barons_train_red
towers_train[0::2]= towers_train_blue - towers_train_red

dragons_train[1::2]= dragons_train_red - dragons_train_blue
barons_train[1::2] = barons_train_red - barons_train_blue
towers_train[1::2]= towers_train_red - towers_train_blue


X_train[:,0] = dragons_train
X_train[:,1] = barons_train
X_train[:,2] = towers_train

print(X_train.shape)

#x_train 데이터와 y_train 데이터로부터 x_test, y_test라는 테스트 데이터를 전체에서 20% 추출
x_train, x_test, y_train, y_test = train_test_split(X_train, y_train, random_state=66, test_size=0.2)


# In[125]:


#세개의 데이터를 벡터 형태로 변환
dragons_train = x_train[:,0]
barons_train = x_train[:,1]
towers_train = x_train[:,2]

print(dragons_train.shape)

dragons_test = x_test[:,0]
barons_test = x_test[:,1]
towers_test = x_test[:,2]

dragons_train = np.asarray(dragons_train).astype('float32').reshape((-1,1))
barons_train = np.asarray(barons_train).astype('float32').reshape((-1,1))
towers_train = np.asarray(towers_train).astype('float32').reshape((-1,1))

dragons_test = np.asarray(dragons_test).astype('float32').reshape((-1,1))
barons_test = np.asarray(barons_test).astype('float32').reshape((-1,1))
towers_test = np.asarray(towers_test).astype('float32').reshape((-1,1))


# In[126]:


#y 테스트 값과 트레인 데이터를 벡터 형태로 변환
y_train = np.asarray(y_train).astype('float32').reshape((-1,1))
y_test = np.asarray(y_test).astype('float32').reshape((-1,1))

print(dragons_train.shape)
print(y_train.shape)

#다중 입력 모델 구현을 위한 input의 shape 형태를 정의
inputA = Input(shape=(None,1))
inputB = Input(shape=(None,1))
inputC = Input(shape=(None,1))


# In[127]:




#inputA는 드래곤 처치수를 입력으로 받는다
#드래곤 처치수에 대한 레이어
x = Dense(64, activation="relu")(inputA)
x = Dense(32, activation="relu")(x)
x = Dense(8, activation="relu")(x)                           
x = Model(inputs=inputA, outputs=x)

#inputB는 바론 처치수를 입력으로 받는다
#바론 처치수를 처리하는 레이어 
y = Dense(64, activation="relu")(inputB)
y = Dense(32, activation="relu")(y)
y = Dense(8, activation="relu")(y)
y = Model(inputs=inputB, outputs=y)
 
#inputC는 타워 철거 수를 입력으로 받는다
z = Dense(64, activation="relu")(inputC)
z = Dense(32, activation="relu")(z)
z = Dense(8, activation="relu")(z)
z = Model(inputs=inputC, outputs=z)
 
    
#x,y,z 각각 모델에 대해 도출된 결과값들을 합친다.
result = concatenate([x.output, y.output,z.output])


#결과값을 바탕으로 한 다중 입력 모델을 설계
k = Dense(2, activation="relu")(result)
#binary classification을 위해 activation function을 sigmoid로 결정
k = Dense(1, activation="sigmoid")(k)


# In[128]:


#train 전체 반복 횟수를 30회
training_epochs = 30 
#일괄 처리 크기를 100으로 설정
batch_size = 100 


#x,y,z에 입력되는 입력값들을 입력으로 받고 output을 k로부터 도출된 결과를 받는 모델 설계
model = Model(inputs=[x.input, y.input,z.input], outputs=k)
print(x.input)

#graident descent 알고리즘을 적용한다.
model.compile(optimizer='sgd', loss = 'binary_crossentropy', metrics=['accuracy'])
#learning rate 0.001로 설정
model.optimizer.lr = 0.001

#모델을 훈련 데이터를 이용해 학습시킨다.
model.fit(x=[dragons_train,barons_train,towers_train], y=y_train, epochs = training_epochs, batch_size=batch_size)


# In[115]:




#설계된 모델을 바탕으로 테스트 데이터를 활용하여 정확도를 계산한다.
evaluation = model.evaluate([dragons_test,barons_test,towers_test], y_test, batch_size=batch_size) 

#정확도가 얼마나 나오는지 출력해본다.
print('Accuracy: ' + str(evaluation[1]))


# In[116]:


def test_accuracy(model):
    evaluation = model.evaluate([dragons_test,barons_test,towers_test], y_test, batch_size=batch_size)  
    print('Accuracy: ' + str(evaluation[1]))


# In[117]:


def save_model(model):
    model.save('LCK.h5')
    


# In[118]:


test_accuracy(model)


# In[119]:


save_model(model)


# In[ ]:




