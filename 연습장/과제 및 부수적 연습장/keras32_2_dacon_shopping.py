from datetime import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings


warnings.filterwarnings('ignore', category=DeprecationWarning)
path = './_data/shopping/'

train = pd.read_csv(path + 'train.csv',
                    index_col=0)

test = pd.read_csv(path + 'test.csv',
                   index_col=0)

add_data = pd.concat([train, test], ignore_index = True)

from pandas_profiling import profile_report

profile = train.profile_report()
print(profile)
