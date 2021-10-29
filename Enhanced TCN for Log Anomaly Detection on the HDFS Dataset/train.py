import json
import pandas as pd
import numpy as np
import keras
from keras.models import Model
from keras.layers import add,Input,Conv1D,Activation,Flatten,Dense,GlobalAveragePooling1D,BatchNormalization
import time
import tensorflow as tf
import keras.backend as K



#Tool 1: Calculate flops
def get_flops(model):
    run_meta = tf.RunMetadata()
    opts = tf.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.profiler.profile(graph=K.get_session().graph,run_meta = run_meta,cmd='op',options=opts)
    return flops.total_float_ops

'''
    Step1:Dimensionality reduction of high-dimensional semantic vectors using PCA-PPA. 
    This step reduces the dimensionality of the 300-dimensional semantic vector into 20-dimensional data.
'''
with open('./data/hdfs_semantic_vec.json') as f:
    # Step1-1 open file
    gdp_list = json.load(f)
    value = list(gdp_list.values())

    # Step1-2 PCA: Dimensionality reduction to 20-dimensional data
    from sklearn.decomposition import PCA
    estimator = PCA(n_components=20)
    pca_result = estimator.fit_transform(value)

    # Step1-3 PPA: De-averaged
    ppa_result = []
    result = pca_result - np.mean(pca_result)
    pca = PCA(n_components=20)
    pca_result = pca.fit_transform(result)
    U = pca.components_
    for i, x in enumerate(result):
        for u in U[0:7]:
            x = x - np.dot(u.transpose(), x) * u
        ppa_result.append(list(x))
    ppa_result = np.array(ppa_result)

'''
    Step2: Read training data.In this process it is necessary to ensure a balance between abnormal and normal samples.
'''
def read_data(path,split = 0.7):
    logs_series = pd.read_csv(path)
    logs_series = logs_series.values
    label = logs_series[:,1]
    logs_data = logs_series[:,0]
    logs = []
    for i in range(0,len(logs_data)):
        padding = np.zeros((300,20))
        data = logs_data[i]
        data = [int(n) for n in data.split()]
        for j in range(0,len(data)):
            padding[j] = ppa_result[data[j]]
        padding = list(padding)
        logs.append(padding)
    logs = np.array(logs)
    split_boundary = int(logs.shape[0] * split)
    train_x = logs[: split_boundary]
    valid_x = logs[split_boundary:]
    train_y = label[: split_boundary]
    valid_y = label[split_boundary:]
    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 20))
    valid_x = np.reshape(valid_x, (valid_x.shape[0], valid_x.shape[1], 20))
    train_y = keras.utils.to_categorical(np.array(train_y))
    valid_y = keras.utils.to_categorical(np.array(valid_y))
    return train_x, train_y, valid_x, valid_y

# Residual block
def ResBlock(x,filters,kernel_size,dilation_rate):
    r = Conv1D(filters,kernel_size,padding='same',dilation_rate=dilation_rate,activation='relu')(x)
    # Reparameterization
    # r = BatchNormalization()(r)/Weight Normalization(r)
    r = Conv1D(1,3,padding='same',dilation_rate=dilation_rate)(r)
    # r = BatchNormalization()(r)/Weight Normalization(r)
    if x.shape[-1]==filters:
        shortcut=x
    else:
        shortcut=Conv1D(filters,kernel_size,padding='same')(x)  #shortcut
    o=add([r,shortcut])
    o=Activation('relu')(o)
    return o

'''
    Step3: training model.Since our proposed method works better, we have not optimised the parameters too much.
    This method can be further improved by parameter optimisation.
'''
def TCN(train_x,train_y,valid_x,valid_y):
    inputs=Input(shape = (300,20))
    x=ResBlock(inputs,filters=3,kernel_size=3,dilation_rate=1)
    x=ResBlock(x,filters=3,kernel_size=3,dilation_rate=2)
    x=ResBlock(x,filters=3,kernel_size=3,dilation_rate=4)
    x=ResBlock(x,filters=3,kernel_size=3,dilation_rate=8)
    # x=Flatten()(x)
    x=GlobalAveragePooling1D()(x)
    x=Dense(2,activation='softmax')(x)
    model=Model(input=inputs,output=x)
    print('flops is ',get_flops(model))
    #查看网络结构
    model.summary()
    #编译模型
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    #训练模型
    model.fit(train_x,train_y,batch_size=64,nb_epoch=100,verbose=2,validation_data=(valid_x,valid_y))
    model_path = './model/E_TCN_GAP.h5'
    model.save(model_path)

# Path
train_path = './data/log_train.csv'
# Training data and valid data
train_x,train_y,valid_x,valid_y = read_data(train_path)
# Training
TCN(train_x,train_y,valid_x,valid_y)

