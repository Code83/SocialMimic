import numpy as np
import pandas as pd
import random
import math,time,sys
from matplotlib import pyplot
from datetime import datetime


df = pd.read_csv(sys.argv[1])
df = df.drop(df.columns[0], axis=1)
(row,col)=np.shape(df)
# print(row,col)
data = df.values

for i in range(row):
	# i = 0
	curr = data[i,:]
	# print(curr)
	temp = set(curr)
	# print(temp)
	temp = np.array(list(temp))
	# print(temp)
	temp.sort()
	# print(temp)
	temp1 = temp#[::-1]
	temp = list(temp1)
	inxList=[]
	for j in curr:
		inx = temp.index(j)
		inxList.append(inx)
		print(inx+1,end=',')
	# print(inxList)
	print()