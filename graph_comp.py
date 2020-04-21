import pandas as pd
import matplotlib.pyplot as plt 
import sys, random, math
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize

# from google.colab import files

# !rm graph_ncaa.csv
# uploaded = files.upload()

df = pd.read_csv('average.csv')
print(list(df))
headers = list(df)

datasetList = df['y'].values
print(datasetList)
df=df.drop(df.columns[0], axis=1)
col = np.shape(df)[1]
xAxis = [i+1 for i in range(col)]
tickLabel = list(df)

colorRow = [ '#800000', '#9A6324', '#469990', '#000075', '#808000', '#0a0a0a', '#bfef45', '#911eb4', '#e6beff','#f032e6',
'#42d4f4', '#e6194B','#ffe119', '#f58231',  '#4363d8', '#3cb44b' ]
# print(np.shape(colorRow))
# plt.show()
i = 0
# for i in range(18):
row = df.values[i,:]
print(row)
print(xAxis)
plt.figure(figsize=(9,6)) 
plt.bar(x=xAxis, height=row, tick_label = tickLabel, width = 1, color=colorRow)  #,  color = ['red', 'green']
plt.xticks(fontsize=16, rotation=60)
lowLimit = 0.88 #min(row)
# if (lowLimit - math.floor(lowLimit)) > 0.5 :
# 	lowLimit = math.floor(lowLimit)
# else:
# 	lowLimit = math.floor(lowLimit) - 1
upLimit = max(row) #min(100,math.ceil(max(row)))
plt.ylim([lowLimit,upLimit])
print(datasetList[i],lowLimit,upLimit)
plt.ylabel('Accuracy',fontsize = 16)
plt.title(datasetList[i],fontsize = 24)
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),fancybox=True, shadow=True, ncol=6)
# # plt.show()
name = 'avgAcc.png' #+ datasetList[i] +
plt.savefig(name, bbox_inches='tight')
