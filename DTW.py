
def min3(a,b,c):
	if (a < b and a < c):
		return a
	elif(b < c):
		return b
	return c

def euclidean(vec1,vec2):
	dist=0
	for i in range(len(vec1)):
		dist+=(vec1[i] - vec2[i])**2
	return dist**(1/2.0)

#pass 2 2D matrices and it will return the DTW distance between them(two sequences)
def dtw_dist(mat1,mat2):
	row = len(mat1)
	col = len(mat2)
	dtw = [[euclidean(mat1[i],mat2[j]) for j in range(col)] for i in range(row)]
	for i in range(1,col):
		dtw[0][i] +=dtw[0][i-1]
	for i in range(1,row):
		dtw[i][0] +=dtw[i-1][0]
	#r=abs(row-col)
	for i in range(1,row):
		for j in range(1,col):
			#if i>j+r or i < j-r
			dtw[i][j] = min3(dtw[i][j-1],dtw[i-1][j],dtw[i-1][j-1]) + dtw[i][j]
	return (1.0/(row*col))*dtw[row-1][col-1]

# A=[[-0.87,-0.88],[-0.84,-0.91],[-0.85,-0.84],[-0.82,-0.82],[-0.23,-0.24],[1.95,1.92],[1.36,1.41],[0.60,0.51],[0.0,0.03],[-0.29,-0.18]]
# B=[[-0.60,-0.46],[-0.65,-0.62],[-0.71,-0.68],[-0.58,-0.63],[-0.17,-0.32],[0.77,0.74],[1.94,1.97]]
# print(dtw_dist(A,B))
