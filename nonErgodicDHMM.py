import numpy as np
from PIL import Image
import os
import random
import math
from operator import itemgetter
import DTW

patchHeight = 64
patchWidth = 64
bins = 8
N = 7
M = 5

dataTrain=[]
dataTest=[]
symbol_Train=[]
symbol_Test=[]
Lambda=[]

def maxi_index(a,b,c):
	if (a > b and a > c):
		return 0
	elif(b > c):
		return 1
	return 2


def calcVectorforPatch(img,xPnt,yPnt):
	vector_ = [0 for i in range(3*bins)]
	for i in range(patchHeight):
		for j in range(patchWidth):
			vector_[int(img[xPnt+i][yPnt+j][0]/32)]+=1;
			vector_[int(8+img[xPnt+i][yPnt+j][1]/32)]+=1;
			vector_[int(16+img[xPnt+i][yPnt+j][2]/32)]+=1;
	return vector_



#create histograms of all the patches and store them into a file in folder output
def createHistograms(folder, patchHeight, patchWidth, bins):
	for filename in os.listdir(folder):
		if os.path.splitext(filename)[1] == ".jpg":
			img = Image.open(os.path.join(folder,filename))
			if img is not None:
				filename = os.path.splitext(filename)[0]
				filename+='.txt'
				img=np.array(img)
				z = img.shape
				x=z[0]
				y=z[1]
				xPnt=0
				with open(os.path.join(folder,filename), 'w') as outfile:
					for i in range(int(x/patchHeight)):
						yPnt=0
						for j in range(int(y/patchWidth)):
							vect = calcVectorforPatch(img,xPnt,j*patchWidth)
							outfile.write(' '.join(str(e) for e in vect))
							outfile.write("\n")
						xPnt+=patchHeight
				outfile.close()
		





def loadData(folder):
	data=[]
	for filename in os.listdir(folder):
		if os.path.splitext(filename)[1] == ".txt" or os.path.splitext(filename)[1] == ".mfcc":
			temp=[]
			with open(os.path.join(folder,filename), 'r') as outfile:
				for line in outfile:
					temp.append([float(v) for v in line.split()])
			data.append(temp)
	return data


def calcDist(x,y):
	distance=0
	z=len(x)
	for i in range(z):
		distance+=((x[i]-y[i])**2)
	return math.sqrt(distance)


#assuming that number of datapoints we have are greater than number of clusters we want to form
def K_MeansClustering(dataMatrix,K):
	
	dimension = len(dataMatrix[0][0][0])
	#	Assigning random means to the K clusters...
	tempClusterMean=[[0 for i in range(dimension)] for i in range(K)]
	#randomKMeans=random.sample(range(0,N),K%N)
	for i in range(K):
		for j in range(dimension):
			tempClusterMean[i][j]=dataMatrix[i%len(dataMatrix)][i%len(dataMatrix[i%len(dataMatrix)])][i%len(dataMatrix[i%len(dataMatrix)][i%len(dataMatrix[i%len(dataMatrix)])])][j]

	#	Dividing the data of this class to K clusters...
	symbol=[[[-1 for k in range(len(dataMatrix[i][j]))] for j in range(len(dataMatrix[i]))] for i in range(len(dataMatrix))]
	tempClusters=[[] for i in range(K)]
	totDistance=0
	for i in range(len(dataMatrix)):
		for j in range(len(dataMatrix[i])):
			for k in range(len(dataMatrix[i][j])):
				minDist=np.inf
				minDistInd=0
				for l in range(K):
					Dist=calcDist(dataMatrix[i][j][k],tempClusterMean[l])
					if Dist<minDist:
						minDist=Dist
						minDistInd=l
				symbol[i][j][k]=minDistInd
				tempClusters[minDistInd].append(dataMatrix[i][j][k])
				totDistance+=minDist

	#	Re-evaluating centres until the energy of changes becomes insignificant...
	energy=100
	while energy>60:
		tempClusterMean=[[0 for i in range(dimension)] for i in range(K)]
		for i in range(K):
			for j in range(len(tempClusters[i])):
				for k in range(dimension):
					tempClusterMean[i][k]+=tempClusters[i][j][k]
			for k in range(dimension):
				if len(tempClusters[i])==0:
					#tempClusterMean[i]=
					break;
				else:
					tempClusterMean[i][k]/=len(tempClusters[i])
		tempClusters=[[] for i in range(K)]
		newTotDistance=0

		for i in range(len(dataMatrix)):
			for j in range(len(dataMatrix[i])):
				for k in range(len(dataMatrix[i][j])):
					minDist=np.inf
					minDistInd=0
					for l in range(K):
						Dist=calcDist(dataMatrix[i][j][k],tempClusterMean[l])
						if Dist<minDist:
							minDist=Dist
							minDistInd=l
					symbol[i][j][k]=minDistInd
					tempClusters[minDistInd].append(dataMatrix[i][j][k])
					newTotDistance+=minDist

		energy=math.fabs(totDistance-newTotDistance);
		totDistance=newTotDistance;
		#print("KNN",energy,len(tempClusters[0]),len(tempClusters[1]),len(tempClusters[2]))
	#return tempClusterMean,symbol
	return symbol

#it should return the probability of a observation sequence given a Lambda[pie,A,B]
def prob(sequence,symbol,lambda_):
	new = [lambda_[0][i]*lambda_[2][i][symbol[0]] for i in range(N)]
	old = [0 for i in range(N)]
	for i in range(1,len(sequence)):
		old=new
		new=[0 for z in range(N)]
		for j in range(N):
			for k in range(N):
				new[j]+=old[k]*lambda_[1][k][j]
			new[j]*=lambda_[2][j][symbol[i]]

	probab=0
	for i in range(len(new)):
		probab+=new[i]

	return probab
	# leng=len(sequence)
	# alpha=[[[0,0] for j in range(leng)] for i in range(N)]
	# for i in range(N):
	# 	alpha[i][0][0] = lambda_[0][i]*lambda_[2][i][symbol[0]]

	# for i in range(1,leng):
	# 	for j in range(N):
	# 		for k in range(N):
	# 			alpha[j][i][0]+=alpha[k][i-1][0]*lambda_[1][k][j]
	# 		alpha[j][i][0]*=lambda_[2][j][symbol[i]]

	
	# for i in range(N):
	# 	alpha[i][leng-1][1] = 1


	# for i in range(1,leng):
	# 	for j in range(N):
	# 		for k in range(N):
	# 			alpha[j][leng-i-1][1]+=lambda_[1][j][k]*lambda_[2][k][symbol[leng-i]]*alpha[k][leng-i][1]
	# probab=0
	# for i in range(N):
	# 	probab+=alpha[i][leng-1][0]

	# return probab



#this will take all the input sequence of a class,symbols of the class, lambda_ and returns 3d matrix of alpha and beta at various time for all sequence
def preProcess(sequence,symbol,lambda_):
	leng=len(sequence)
	alpha=[[[0,0] for j in range(leng)] for i in range(N)]
	for i in range(N):
		alpha[i][0][0] = lambda_[0][i]*lambda_[2][i][symbol[0]]

	for i in range(1,leng):
		for j in range(N):
			for k in range(N):
				alpha[j][i][0]+=alpha[k][i-1][0]*lambda_[1][k][j]
			alpha[j][i][0]*=lambda_[2][j][symbol[i]]

	
	for i in range(N):
		alpha[i][leng-1][1] = 1


	for i in range(1,leng):
		for j in range(N):
			for k in range(N):
				alpha[j][leng-i-1][1]+=lambda_[1][j][k]*lambda_[2][k][symbol[leng-i]]*alpha[k][leng-i][1]

	return alpha



#it should return the list [A,B,pie] or HMM parameters of the class
def generateHMM(data,symbol):
	#initialising parameters
	pie = [1.0/N for i in range(N)]
	A = [[1.0/(N-i) for j in range(N)] for  i in range(N)]
	for i in range(N):
		for j in range(i+1,N):
			A[j][i]=0

	B = [[1.0/M for j in range(M)] for i in range(N)]
	lambda_old=[pie,A,B]
	p_old=p_new=0
	for i in range(len(data)):
		p_new+=prob(data[i],symbol[i],lambda_old)

	error=100
	while error>1e-3:
		pie = [0 for i in range(N)]
		A = [[0 for j in range(N)] for  i in range(N)]
		B = [[0 for j in range(M)] for i in range(N)]
		lambda_new=[pie,A,B]

		#re-estimating parameters
		for i in range(len(data)):
			#N*L*2 matrix
			alpha = preProcess(data[i],symbol[i],lambda_old)
			prob_O = prob(data[i],symbol[i],lambda_old)

			#re-estimating pie
			tempPie=[0 for l in range(N)]
			for j in range(N):
				for k in range(N):
					tempPie[j]+=alpha[j][0][0]*lambda_old[1][j][k]*lambda_old[2][k][symbol[i][1]]*alpha[k][1][1]
				if prob_O != 0:
					tempPie[j]/=prob_O
				else:
					break

			for j in range(N):
				lambda_new[0][j]+=tempPie[j]
			

			tempAi=[0 for j in range(N)]
			#re-estimating A
			for j in range(N):
				tempA=[0 for l in range(N)]
				for k in range(j,N):
					for t in range(len(data[i])-1):
						tempA[k]+=alpha[j][t][0]*lambda_old[1][j][k]*lambda_old[2][k][symbol[i][t+1]]*alpha[k][t+1][1]
			
				temp=0
				for t in range(len(data[i])-1):
					temp+=alpha[j][t][0]*alpha[j][t][1]
				for k in range(N):
					if temp !=0:
						lambda_new[1][j][k]+=tempA[k]/temp
					else:
						#print("i am in A")
						lambda_new[1][j][k]+=0
				#tempAi[j]=temp


			#re-estimating B
			for j in range(N):
				tempB=[0 for k in range(M)]
				for k in range(M):
					for t in range(len(data[i])-1):
						if symbol[i][t]==k:
							tempB[k]+=alpha[j][t][0]*alpha[j][t][1]

				temp_=0
				for t in range(len(data[i])-1):
					temp_+=alpha[j][t][0]*alpha[j][t][1]
				
				for k in range(M):
					if(temp_!=0):
						lambda_new[2][j][k]+=(tempB[k]/temp_)
					else:
						#print("i am in B")
						lambda_new[2][j][k]+=0

		#averaging on all the sequences
		for i in range(N):
			lambda_new[0][i]/=len(data)
		for i in range(N):
			for j in range(N):
				lambda_new[1][i][j]/=len(data)
		for i in range(N):
			for j in range(M):
				lambda_new[2][i][j]/=len(data)
		#done re-estimating parameters

		p_old=p_new
		p_new=0
		lambda_old=lambda_new
		for i in range(len(data)):
			z=prob(data[i],symbol[i],lambda_new)
			if z<=0:
				continue
			else:
				p_new+=math.log(z)
		error=math.fabs(p_new - p_old)
		#print ("error",error)
	return lambda_new


dataset = input("Which dataset you want to use:('i' for image and 's' for speech): ")
N=int(input("Input the number of states: "))
M=int(input("Input the number of discrete observation symbols: "))
#loading train data
if dataset=='i':

	folder = "D:\\Sem 5\\CS 669\\HMM\\Group02\\image\\Train"
	for foldername in os.listdir(folder):
		createHistograms(os.path.join(folder,foldername),patchHeight,patchWidth,bins)
		temp = loadData(os.path.join(folder,foldername))
		dataTrain.append(temp)
		temp=[]
	#concat(folder)
	print("Training data loaded successfully")
	symbol_Train = K_MeansClustering(dataTrain,M)

elif dataset=='s':

	folder= "D:\\Sem 5\\CS 669\\HMM\\Group02\\Text\\Train"
	for foldername in os.listdir(folder):
		temp = loadData(os.path.join(folder,foldername))
		dataTrain.append(temp)
		temp=[]
	#concat(folder)
	print("Training data loaded successfully")
	symbol_Train = K_MeansClustering(dataTrain,M)



#generating HMM for all classes
for i in range(len(dataTrain)):
	#print("generating HMM for")
	#print(i)
	temp = generateHMM(dataTrain[i],symbol_Train[i])
	Lambda.append(temp)
print("Done making HMM for all the classes")

print("Loading test data")
#loading test data
if dataset=='i':

	folder = "D:\\Sem 5\\CS 669\\HMM\\Group02\\image\\Test"
	for foldername in os.listdir(folder):
		createHistograms(os.path.join(folder,foldername),patchHeight,patchWidth,bins)
		temp = loadData(os.path.join(folder,foldername))
		dataTest.append(temp)
		temp=[]
	#concat(folder)
	print("Test data loaded successfully")
	symbol_Test = K_MeansClustering(dataTest,M)


elif dataset=='s':

	folder = "D:\\Sem 5\\CS 669\\HMM\\Group02\\Text\\Test"
	for foldername in os.listdir(folder):
		temp = loadData(os.path.join(folder,foldername))
		dataTest.append(temp)
		temp=[]
	#concat(folder)
	print("Test data loaded successfully")
	symbol_Test = K_MeansClustering(dataTest,M)

#calculating confusion matrix
confusionMatrix=[[0 for j in range(len(dataTrain))] for i in range(len(dataTrain))]

for i in range(len(dataTest)):
	for j in range(len(dataTest[i])):
		xx=prob(dataTest[i][j],symbol_Test[i][j],Lambda[0])
		yy=prob(dataTest[i][j],symbol_Test[i][j],Lambda[1])
		zz=prob(dataTest[i][j],symbol_Test[i][j],Lambda[2])
		ind = maxi_index(xx,yy,zz)
		confusionMatrix[ind][i]+=1

print(confusionMatrix)

Nr=Dr=0
for i in range(len(confusionMatrix)):
	for j in range(len(confusionMatrix[0])):
		if i==j:
			Nr+=confusionMatrix[i][j]
		Dr+=confusionMatrix[i][j]
print("Accuracy: ",Nr/Dr)

meanPrecision=0
meanRecall=0
meanFmeasure=0
x=y=z=0
for i in range(len(confusionMatrix)):
	Dr=0
	for j in range(len(confusionMatrix[i])):
		Dr+=confusionMatrix[i][j]
	if Dr!=0:
		x=confusionMatrix[i][i]/Dr
		meanPrecision+=x
		print("Precision for class ",i,": ",x)
	Dr=0
	for j in range(len(confusionMatrix)):
		Dr+=confusionMatrix[j][i]
	if Dr!=0:
		y=confusionMatrix[i][i]/Dr
		meanRecall+=y
		print("Recall for class ",i,": ",y)
	if (x+y)!=0:
		z=2*x*y/(x+y)
		meanFmeasure+=z
		print("F-measure for class ",i,"; ",z)

print("Mean Precision: ",meanPrecision/len(confusionMatrix))
print("Mean Recall: ",meanRecall/len(confusionMatrix))
print("Mean F-measure: ",meanFmeasure/len(confusionMatrix))
