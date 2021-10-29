import pandas as pd
import numpy as np
np.set_printoptions(threshold=np.inf)
import csv
import copy
from collections import Counter

pre_data = pd.read_csv('./bgl/BGL_sequence.csv')
pre_data = pre_data.values

data = []
label = []
count = 0
co = 0
for i in range(0,len(pre_data)):
    value = []
    division = pre_data[i][0].split(",")
    if division[0] != '[]':
        for j in range(0,len(division)):
            if '[' in division[j] and ']' not in division[j]:
                value.append(int(division[j][3:-1]))
            elif '[' in division[j] and ']' in division[j]:
                value.append(int(division[j][3:-2]))
            elif '[' not in division[j] and ']' in division[j]:
                value.append(int(division[j][3:-2]))
            else:
                value.append(int(division[j][3:-1]))
        line = str(np.array(value))[2:-1].split(' ')
        data.append(str(np.array(value))[1:-1])
        label.append(int(pre_data[i][1]))
        co = co+1
    else:
        count = count+1
print(co)
print(count)

pd.DataFrame({'Sequence':data,'label':label}).to_csv('./data/bgl_time_data.csv',index=False, header=False)
