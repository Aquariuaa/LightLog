import json
import pandas as pd
import numpy as np
import keras
from keras.models import load_model
import time




with open('./data/bgl_semantic_vec.json') as f:
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

def read_test(split = 0.7):
    logs_data = pd.read_csv('./data/bgl_data.csv')
    logs_data = logs_data.values
    label = pd.read_csv('./data/bgl_label.csv')
    label = label.values
    logs = []
    for i in range(0,len(logs_data)):
        padding = np.zeros((300,20))
        data = logs_data[i]
        for j in range(0,len(data)):
            padding[j] = pca_result[int(data[j]-1)]
        padding = list(padding)
        logs.append(padding)
    logs = np.array(logs)
    split_boundary = int(logs.shape[0] * split)
    valid_x = logs[split_boundary:]
    test_y = label[split_boundary:]
    valid_x = np.reshape(valid_x, (valid_x.shape[0],valid_x.shape[1],20))
    valid_y = keras.utils.to_categorical(np.array(test_y))
    return valid_x,valid_y,test_y



def test_model(test_x):
    model = load_model('./model/E-TCN.h5')
    y_pred = model.predict(test_x, batch_size=512)
    return y_pred


test_x,valid_y,label = read_test()
start = time.clock()
y_pred = test_model(test_x)
end = time.clock()
print('The detection time is',end-start)

y_pred = np.argmax(y_pred, axis=1)
tp = 0
fp = 0
tn = 0
fn = 0
for j in range(0, len(y_pred)):
    if label[j] == y_pred[j] and label[j] == 0:
        tp = tp + 1
    elif label[j] != y_pred[j] and label[j] == 0:
        fp = fp + 1
    elif label[j] == y_pred[j] and label[j] == 1:
        tn = tn + 1
    elif label[j] != y_pred[j] and label[j] == 1:
        fn = fn + 1

print('TP,FP,TN,FN are: ',[tp,fp,tn,fn])
print('Precision, Recall, F1-measure are:',tn/(tn+fn),tn/(tn+fp),2*(tn/(tn+fn)*(tn/(tn+fp))/(tn/(tn+fn)+tn/(tn+fp))))
datas = pd.DataFrame(data=[tp,fp,tn,fn])
datas.to_csv('./result/result_BGL',index=False,header=False)
