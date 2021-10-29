import pandas as pd
import numpy as np
from collections import Counter

test = pd.read_csv('./robust_log_test.csv')
test = test.values

np.random.shuffle(test)
np.random.shuffle(test)
np.random.shuffle(test)
np.random.shuffle(test)

test = test[0:50000]
label = Counter(test[:,1])
print(label)
save_test = pd.DataFrame(data=test)
save_test.to_csv('./rubust_log_test_50000.csv',index=False,header=False)


