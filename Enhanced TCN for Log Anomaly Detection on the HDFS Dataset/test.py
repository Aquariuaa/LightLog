import json
import pandas as pd
import numpy as np
import keras
from keras.models import load_model
import time

'''
    This step reduces the dimensionality of the 300-dimensional semantic vector into 20-dimensional data. 
    Due to the overwhelming size of the original dataset, a sample of 2000 was selected for presentation.
'''

with open('./data/hdfs_semantic_vec.json') as f:
    gdp_list = json.load(f)
    value = list(gdp_list.values())
    from sklearn.decomposition import PCA
    estimator = PCA(n_components=20)
    pca_result = estimator.fit_transform(value)
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

# test data
def read_test(path):
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
    train_x = logs
    train_y = label
    text_x = np.reshape(train_x, (train_x.shape[0],train_x.shape[1],20))
    text_y = keras.utils.to_categorical(np.array(train_y))
    return text_x,text_y,label


def test_model(test_x):
    model = load_model('./model/E_TCN_GAP.h5')
    y_pred = model.predict(test_x, batch_size=512)
    return y_pred

# Data handling
start_data = time.clock()
test_path = './data/log_test_2000.csv'
test_x,test_y,label = read_test(test_path)
end_data = time.clock()
print('The data processing time is',end_data-start_data)

# Detect
start_detect = time.clock()
y_pred = test_model(test_x)
end_detect = time.clock()
print('The detection time is',end_detect-start_detect)

# Input result
y_pred = np.argmax(y_pred,axis=1)
tp=0
fp=0
tn=0
fn=0
for j in range(0,len(y_pred)):
    if label[j] == y_pred[j] and label[j]==0:
        tp=tp+1
    elif label[j]!=y_pred[j] and label[j]==0:
        fp=fp+1
    elif label[j]==y_pred[j] and label[j]==1:
        tn=tn + 1
    elif label[j]!=y_pred[j] and label[j]==1:
        fn=fn + 1

print('TP,FP,TN,FN are: ',[tp,fp,tn,fn])
print('Precision, Recall, F1-measure are:',tn/(tn+fn),tn/(tn+fp),2*(tn/(tn+fn)*(tn/(tn+fp))/(tn/(tn+fn)+tn/(tn+fp))))
datas = pd.DataFrame(data=[tp,fp,tn,fn])
datas.to_csv('./result/result_HDFS',index=False,header=False)



