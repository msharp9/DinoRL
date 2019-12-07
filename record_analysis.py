import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RECORD = "records/record.txt"

df = pd.read_csv(RECORD, names=['model', 'timestamp', 'reward', 'time',
    'steps', 'avg_q_max', 'avg_loss'])
df['timestamp'] = pd.to_datetime(df['time'],unit='s')
print(df.head())

# df.plot(y='avg_q_max', use_index=True)

xaxis = ['index']
yaxis = ['reward', 'time', 'steps', 'avg_q_max', 'avg_loss']
for x in xaxis:
    for y in yaxis:
        fig, ax = plt.subplots(nrows=1, ncols=1)
        fig.suptitle(y)
        ax.set_ylabel(y)
        df.reset_index().plot.scatter(x=x, y=y, ax=ax)
        plt.show()

# xaxis = ['time']
# yaxis = ['reward', 'steps', 'avg_q_max', 'avg_loss']
# for x in xaxis:
#     for y in yaxis:
#         fig, ax = plt.subplots(nrows=1, ncols=1)
#         fig.suptitle(y)
#         ax.set_ylabel(y)
#         df.reset_index().plot(x=x, y=y, ax=ax)
#         plt.show()
#
# yaxis = ['reward', 'steps', 'avg_q_max', 'avg_loss']
# for x in yaxis:
#     for y in yaxis:
#         if x != y:
#             fig, ax = plt.subplots(nrows=1, ncols=1)
#             fig.suptitle(y)
#             ax.set_ylabel(y)
#             df.reset_index().plot.scatter(x=x, y=y, ax=ax)
#             plt.show()
