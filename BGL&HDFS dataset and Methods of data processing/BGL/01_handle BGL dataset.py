import pandas as pd
import copy
from collections import Counter

pre_data = pd.read_csv('./bgl/BGL_sequence.csv')
pre_data = pre_data.values

data = []
label = []
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
    else:
        value.append(0)
    data.append(value)
    label.append(int(pre_data[i][1]))
print(Counter(label))

pd.DataFrame(data=data).to_csv('../data/bgl_data.csv', index=False, header=False)
pd.DataFrame(data=label).to_csv('../data/bgl_label.csv', index=False, header=False)
