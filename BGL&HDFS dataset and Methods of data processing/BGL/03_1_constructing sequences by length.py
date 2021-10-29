import pandas as pd
import numpy as np
from collections import Counter

sequence_length = 300
data = pd.read_csv('./bgl/BGL_100k_structured.csv')
data = data.values
pre_label = data[:,0]
logs = data[:,2]
# 处理datas
for i in range(0,len(logs)):
    logs[i] = int(logs[i][1:])
#处理label
label = []
for l in range(0,len(pre_label)):
    if pre_label[l]=='-':
        label.append(0)
    else:
        label.append(1)

logs_data = []
for j in range(len(logs) - sequence_length):
    logs_data.append(logs[j: j + sequence_length])
reshaped_logs = np.array(logs_data).astype('float64')

logs_label = []
for k in range(len(label) - sequence_length):
    logs_label.append(label[k: k + sequence_length])
reshaped_label = np.array(logs_label).astype('float64')
# reshaped_label = logs_label

result_label = []
for m in range(0,len(reshaped_label)):
    if 1 in reshaped_label[m]:
        result_label.append(1)
    else:
        result_label.append(0)
end_logs = []
end_label = []
for n in range(0,len(result_label)):
    # if n%10 == 0:
    end_logs.append(reshaped_logs[n])
    end_label.append(result_label[n])
print(Counter(end_label))

pd.DataFrame(data=end_logs).to_csv('../data/bgl_data.csv', index=False, header=False)
pd.DataFrame(data=end_label).to_csv('../data/bgl_label.csv', index=False, header=False)