import numpy as np
import pandas as pd
import random
import math,time,sys, os
from matplotlib import pyplot
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

#==================================================================
def sigmoid1(gamma):     #convert to probability
	if gamma < 0:
		return 1 - 1/(1 + math.exp(gamma))
	else:
		return 1/(1 + math.exp(-gamma))

def sigmoid1i(gamma):     #convert to probability
	gamma = -gamma
	if gamma < 0:
		return 1 - 1/(1 + math.exp(gamma))
	else:
		return 1/(1 + math.exp(-gamma))

def sigmoid2(gamma):
	gamma /= 2
	if gamma < 0:
		return 1 - 1/(1 + math.exp(gamma))
	else:
		return 1/(1 + math.exp(-gamma))
		
def sigmoid3(gamma):
	gamma /= 3
	if gamma < 0:
		return 1 - 1/(1 + math.exp(gamma))
	else:
		return 1/(1 + math.exp(-gamma))

def sigmoid4(gamma):
	gamma *= 2
	if gamma < 0:
		return 1 - 1/(1 + math.exp(gamma))
	else:
		return 1/(1 + math.exp(-gamma))


def Vfunction1(gamma):
	return abs(np.tanh(gamma))

def Vfunction2(gamma):
	val = (math.pi)**(0.5)
	val /= 2
	val *= gamma
	val = math.erf(val)
	return abs(val)

def Vfunction3(gamma):
	val = 1 + gamma*gamma
	val = math.sqrt(val)
	val = gamma/val
	return abs(val)

def Vfunction4(gamma):
	val=(math.pi/2)*gamma
	val=np.arctan(val)
	val=(2/math.pi)*val
	return abs(val)

def x1copy(gamma):
	s1 = abs(gamma)*0.5 + 1
	s1 = (-gamma)/s1 + 0.5
	return s1

def x2copy(gamma):
	s2 = abs(gamma - 1)*0.5 + 1
	s2 = (gamma - 1)/s2 + 0.5
	return s2

def initialize(popSize,dim):
	population=np.zeros((popSize,dim))
	minn = 1
	maxx = math.floor(0.8*dim)
	if maxx<minn:
		minn = maxx
	
	for i in range(popSize):
		random.seed(i**3 + 10 + time.time() ) 
		no = random.randint(minn,maxx)
		if no == 0:
			no = 1
		random.seed(time.time()+ 100)
		pos = random.sample(range(0,dim-1),no)
		for j in pos:
			population[i][j]=1
		
		# print(population[i])  
	return population

def fitness(solution, trainX, testX, trainy, testy):
	cols=np.flatnonzero(solution)
	val=1
	if np.shape(cols)[0]==0:
		return val	
	clf=KNeighborsClassifier(n_neighbors=5)
	train_data=trainX[:,cols]
	test_data=testX[:,cols]
	clf.fit(train_data,trainy)
	val=1-clf.score(test_data,testy)

	#in case of multi objective  []
	set_cnt=sum(solution)
	set_cnt=set_cnt/np.shape(solution)[0]
	val=omega*val+(1-omega)*set_cnt
	return val

def allfit(population, trainX, testX, trainy, testy):
	x=np.shape(population)[0]
	acc=np.zeros(x)
	for i in range(x):
		acc[i]=fitness(population[i],trainX,testX,trainy,testy)     
		#print(acc[i])
	return acc

def toBinary(currAgent,prevAgent,dimension,trainX,testX,trainy,testy):
	# print("continuous",solution)
	# print(prevAgent)
	
	Xnew = np.zeros(np.shape(currAgent))
	for i in range(dimension):
		temp = Vfunction3(currAgent[i])

		random.seed(time.time()+i)
		# if temp > 0.5: # sfunction
		# 	Xnew[i] = 1
		# else:
		# 	Xnew[i] = 0
		if temp > 0.5: # vfunction
			Xnew[i] = 1 - prevAgent[i]
		else:
			Xnew[i] = prevAgent[i]
	return Xnew


	# Xnew = np.zeros(np.shape(currAgent))
	# Xnew1 = np.zeros(np.shape(currAgent))
	# Xnew2 = np.zeros(np.shape(currAgent))
	# for i in range(dimension):
	# 	temp = sigmoid1(currAgent[i])
	# 	random.seed(time.time()+i)
	# 	r1 = random.uniform(0,1)
	# 	if temp > r1: # sfunction
	# 		Xnew1[i] = 1
	# 	else:
	# 		Xnew1[i] = 0

	# 	random.seed(time.time()+i)
	# 	r1 = random.uniform(0,1)
	# 	temp = sigmoid1i(currAgent[i])
	# 	if temp > r1: # sfunction
	# 		Xnew2[i] = 1
	# 	else:
	# 		Xnew2[i] = 0

	# fit1 = fitness(Xnew1,trainX,testX,trainy,testy)
	# fit2 = fitness(Xnew2,trainX,testX,trainy,testy)
	# fitOld =  fitness(prevAgent,trainX,testX,trainy,testy)
	# if fit1<fitOld or fit2<fitOld:
	# 	if fit1 < fit2:
	# 		Xnew = Xnew1.copy()
	# 	else:
	# 		Xnew = Xnew2.copy()
	# 	return Xnew
	# # else:
	# if fit2<fit1:
	# 	Xnew1 = Xnew2.copy()
	# Xnew3 = Xnew1.copy()
	# Xnew4 = prevAgent.copy()
	# for i in range(dimension):
	# 	random.seed(time.time() + i)
	# 	r2 = random.random()
	# 	if r2>0.6:
	# 		tx = Xnew3[i]
	# 		Xnew3[i] = Xnew4[i]
	# 		Xnew4[i] = tx
	# fit1 = fitness(Xnew3,trainX,testX,trainy,testy)
	# fit2 = fitness(Xnew4,trainX,testX,trainy,testy)
	# if fit1<fit2:
	# 	return Xnew3
	# else:
	# 	return Xnew4
	# ######### print("binary",Xnew)
	# return Xnew


#============================================================================
def socialmimic(dataset,popSize,maxIter,randomstate):

	#--------------------------------------------------------------------
	df=pd.read_csv(dataset)
	(a,b)=np.shape(df)
	print(a,b)
	data = df.values[:,0:b-1]
	label = df.values[:,b-1]
	dimension = np.shape(data)[1] #solution dimension
	#---------------------------------------------------------------------

	cross = 5
	test_size = (1/cross)
	trainX, testX, trainy, testy = train_test_split(data, label,stratify=label ,test_size=test_size,random_state=randomstate) #


	clf=KNeighborsClassifier(n_neighbors=5)
	clf.fit(trainX,trainy)
	val=clf.score(testX,testy)
	whole_accuracy = val
	print("Total Acc: ",val)

	x_axis = []
	y_axis = []
	population = initialize(popSize,dimension)
	GBESTSOL = np.zeros(np.shape(population[0]))
	GBESTFIT = 1000

	start_time = datetime.now()
	
	for currIter in range(1,maxIter):
		newpop = np.zeros((popSize,dimension))
		fitList = allfit(population,trainX,testX,trainy,testy)
		if currIter==1:
			y_axis.append(min(fitList))
		else:
			y_axis.append(min(min(fitList),y_axis[len(y_axis)-1]))
		x_axis.append(currIter)
		bestInx = np.argmin(fitList)
		fitBest = min(fitList)
		Xbest = population[bestInx].copy()

		if fitBest<GBESTFIT:
			GBESTFIT = fitBest
			GBESTSOL = Xbest.copy()
			print("gbest:",GBESTFIT,GBESTSOL.sum())

		for i in range(popSize):
			currFit = fitList[i]
			# print(currFit)
			difference = ( currFit - fitBest ) / currFit
			if difference == 0:
				random.seed(time.time())
				difference = random.uniform(0,1)
			newpop[i] = np.add(population[i],np.multiply(difference,population[i]))
			newpop[i] = toBinary(newpop[i],population[i],dimension,trainX,testX,trainy,testy)

		population = newpop.copy()
	# pyplot.plot(x_axis,y_axis)
	# pyplot.show()

	#test accuracy
	cols = np.flatnonzero(GBESTSOL)
	val = 1
	if np.shape(cols)[0]==0:
		return GBESTSOL
	clf = KNeighborsClassifier(n_neighbors=5)
	train_data = trainX[:,cols]
	test_data = testX[:,cols]
	clf.fit(train_data,trainy)
	val = clf.score(test_data,testy)
	return GBESTSOL,val


#========================================================================================================
popSize = 10
maxIter = 20
omega = 0.9
# datasetList = ["Breastcancer"]
datasetList = ["Breastcancer", "BreastEW", "CongressEW", "Exactly", "Exactly2", "HeartEW", "IonosphereEW", "KrvskpEW", "Lymphography", "M-of-n", "PenglungEW", "SonarEW", "SpectEW", "Tic-tac-toe", "Vote", "WaveformEW", "WineEW", "Zoo"]
randomstateList=[15,5,15,26,12,7,10,8,37,19,35,2,49,26,1,25,47,12]

for datasetinx in range(len(datasetList)):
	dataset=datasetList[datasetinx]
	best_accuracy = -100
	best_no_features = 100
	best_answer = []
	accuList = []
	featList = []
	for count in range(5):
		if (dataset == "WaveformEW" or dataset == "KrvskpEW") and count>2:
			break
		print(count)
		answer,testAcc = socialmimic("csvUCI/"+dataset+".csv",popSize,maxIter,randomstateList[datasetinx])
		print(testAcc,answer.sum())
		accuList.append(testAcc)
		featList.append(answer.sum())
		if testAcc>=best_accuracy and answer.sum()<best_no_features:
			best_accuracy = testAcc
			best_no_features = answer.sum()
			best_answer = answer.copy()
		if testAcc>best_accuracy:
			best_accuracy = testAcc
			best_no_features = answer.sum()
			best_answer = answer.copy()

		
	
	print(dataset,"best:",best_accuracy,best_no_features)
	# inx = np.argmax(accuList)
	# best_accuracy = accuList[inx]
	# best_no_features = featList[inx]
	# print(dataset,"best:",accuList[inx],featList[inx])
	with open("result_SMOv3.csv","a") as f:
		print(dataset,"%.2f"%(100*best_accuracy),best_no_features,sep=',',file=f)
	# with open("result_SMOXarrayA.csv","a") as f:
	# 	print(dataset,end=',',file=f)
	# 	for i in accuList:
	# 		print("%.2f"%(100*i),end=',',file=f)
	# 	print('',file=f)

	# with open("result_SMOXarrayF.csv","a") as f:
	# 	print(dataset,end=',',file=f)
	# 	for i in featList:
	# 		print(int(i),end=',',file=f)
	# 	print('',file=f)
