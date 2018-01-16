
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
		
