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
		


K=5

dataTrain=[]
dataTest=[]


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



def KNN_classLabel(testVec, trainData,K):
	temp=[]
	for i in range(len(trainData)):
		for j in range(len(trainData[i])):
			temp.append([DTW.dtw_dist(testVec,trainData[i][j]),i])
	temp.sort(key=itemgetter(0))
	label=[0 for i in range(len(trainData))]
	for i in range(K):
		label[temp[i][1]]+=1
	return label.index(max(label))



dataset = input("Which dataset you want to use:('i' for image and 's' for speech): ")
K=int(input("Input the value of K: "))

if dataset=='i':

	folder ="D:\\Sem 5\\CS 669\\HMM\\Group02\\image\\Train"
	for foldername in os.listdir(folder):
		createHistograms(os.path.join(folder,foldername),patchHeight,patchWidth,bins)
		temp = loadData(os.path.join(folder,foldername))
		dataTrain.append(temp)
		temp=[]

	folder ="D:\\Sem 5\\CS 669\\HMM\\Group02\\image\\Test"
	for foldername in os.listdir(folder):
		createHistograms(os.path.join(folder,foldername),patchHeight,patchWidth,bins)
		temp = loadData(os.path.join(folder,foldername))
		dataTest.append(temp)
		temp=[]


elif dataset=='s':
	folder ="D:\\Sem 5\\CS 669\\HMM\\Group02\\Text\\Train"
	for foldername in os.listdir(folder):
		temp = loadData(os.path.join(folder,foldername))
		dataTrain.append(temp)
		temp=[]

	folder ="D:\\Sem 5\\CS 669\\HMM\\Group02\\Text\\Test"
	for foldername in os.listdir(folder):
		temp = loadData(os.path.join(folder,foldername))
		dataTest.append(temp)
		temp=[]

#calculating confusion matrix
confusionMatrix=[[0 for j in range(len(dataTrain))] for i in range(len(dataTrain))]

for i in range(len(dataTest)):
	for j in range(len(dataTest[i])):
		confusionMatrix[KNN_classLabel(dataTest[i][j],dataTrain,K)][i]+=1

#print(dataTrain)
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

