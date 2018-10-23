import numpy as np 
from numpy import genfromtxt
from numpy import linalg as LA
import matplotlib.pyplot as plt
import cvxopt as cvxopt
from mpl_toolkits.mplot3d import Axes3D
import random

def separateByclass(x,y,k):
	X= []
	for i in range(0,k):
		t = x[y[:]==i,:]
		X.append(t)
	return X

def statistics(ycalc,ytest,k):
	m = np.zeros((k,k))
	for i in range(0,k):
		ind =  np.where(ytest==i)
		t = ycalc[ind]
		print np.size(t)
		for j in range(0,k):
			t1 = np.where(t==j)
			e = np.size(t1)
			m[i][j]=e

	print m

	return m
	
def PCA(m,k):
	mean = np.mean(m,axis=0)
	m=m-mean
	cov = np.cov(m.transpose())
	eigval , eigvec = LA.eig(cov)
	a = np.argsort(eigval)
	b = eigvec[a]
	comp = b[0:k]
	comp = comp.transpose()
	xtrain1=np.dot(m,comp)
	return xtrain1

def DataVisualization(x):
	plt.figure(0)
	marker=['o','+','^','x','D','*','h','8','p','s','|','_']
	for i in range(0,k):
		plt.scatter(x[i][:,0],x[i][:,1],marker=marker[i])
	plt.show()

def distanceMetric(x1,x2):
	x1=np.asarray(x1)
	x2=np.asarray(x2)
	m=np.size(x1,axis=0)
	n=np.size(x2,axis=0)
	d=np.size(x1,axis=1)
	x1=np.reshape(x1,(1,m,d))
	x2=np.reshape(x2,(n,1,d))
	diff = x1-x2
	diff=diff*diff
	res = np.sum(diff,axis=2)
	return res # every row corresponds to different test sample, every column represents training sample

def calculateAccuracy(ycalc,ytest):
	err = ycalc-ytest
	err = np.where(ycalc==ytest)
	acc = np.size(err)*1.0/np.size(ycalc)*100
	statistics(ycalc,ytest,k)
	return acc

def gaussian(x,mu,cov):
	d=np.size(x,axis=1)
	n=np.size(x,axis=0)
	sigma = cov
	sigmainv=np.linalg.inv(cov)
	x1 = x-mu
	t = np.dot(x1,sigmainv)
	t = np.dot(t,x1.transpose())
	z = t * np.eye(n)
	z = np.sum(z,axis=0)
	z = np.exp(-1*z)
	det = np.linalg.det(cov)
	z = z/(((2*3.14)**(d/2))*(det**(1/2)))
	return z

def perceptron(xtrain,ytrain):
	d = np.size(xtrain,axis=1)
	m=np.size(xtrain,axis=0)
	w = np.zeros((d,1))
	i=0
	count=0
	while(True):
		x = xtrain[i]
		y = ytrain[i]
		i+=1
		if(i==m):
			i=0
			print count,"*****************************"
			count=0
		x=x[:,None]
		z = np.dot(x.T,w)
		z=(z>0)*1
		if(z==0 and y==1):
			c=1
		elif(z==1 and y==0):
			c=-1
		else:
			c=0
		if c==0:
			count=count+1
			if(count>0.75*m):
				break
		else:
			w = w + c*x
	

	return w

def perceptronOneVsAll(xtrain,ytrain,xtest,ytest):
	Ytrain = sepDataForOneVsAll(ytrain,k)
	m=np.size(xtrain,axis=0)
	b = np.ones((m,1))
	xtrain = np.hstack((b,xtrain))
	Z=np.zeros((k,m))
	for i in range(0,k):
		w = perceptron(xtrain,Ytrain[i])
		z = np.dot(xtrain,w)
		Z[i][:,None]=z
	Z=Z.T
	ycalc = np.argmax(Z,axis=1)
	acc = calculateAccuracy(ycalc,ytrain)
	print "Accuracy is ",acc



def addPolynomialFeatures(x,p):
	x=x[:,None]
	a = np.copy(x)
	X = np.copy(x)
	for i in range(2,p+1):
		a = a*x
		X = np.hstack((X,a))
	return X

def Visualiztion3D(x):
	fig=plt.figure()
	color=['red','blue','green','yellow','orange','purple','black','pink','aqua']
	ax = fig.add_subplot(111, projection='3d')
	for i in range(0,k):


		ax.scatter(x[i][:,0],x[i][:,1],x[i][:,2],color=color[i])
	plt.show()

def oneHot(y,k):
	m = np.size(y,axis=0)
	y=np.squeeze(y)
	b=np.arange(m)
	Y = np.zeros((m,k))
	Y[b,y.astype(int)]=1
	return Y

def MultiClassLinearModel(xtrain,ytrain,x,step,xtest,ytest):

	d = np.size(xtrain,axis=1)
	m = np.size(xtrain,axis=0)
	b = np.ones((m,1))
	xtrain = np.hstack((b,xtrain))
	ytrain = oneHot(ytrain,k)
	w=np.random.rand(k,d+1)

	fig = plt.figure()
	i=0
	lprev=10000
	while(True):
		z = np.dot(xtrain,w.transpose())
		l=MSELoss(z,ytrain)/m
		l=np.sum(l)
		if(abs(lprev-l)<0.00001):
			break
		plt.scatter(i,l,color='black',s=2)
		g = MSEGradient(z,ytrain,xtrain)/m
		w=w-step*g
		lprev=l
		i+=1

	name = "LR_"+str(step)+".png"
	plt.savefig(name)
	plt.clf
	acc = testing(xtest,ytest,w)
	return acc

def LinearRegModel(xtrain,ytrain,xtest,ytest):
	d = np.size(xtrain,axis=1)
	m = np.size(xtrain,axis=0)
	m1 = np.size(xtest,axis=0)
	b = np.ones((m,1))
	b1 = np.ones((m1,1))
	xtrain = np.hstack((b,xtrain))
	xtest = np.hstack((b1,xtest))
	w=NormalEqnSoln(xtrain,ytrain)
	z = np.dot(xtest,w)
	err = ytest[:,None] - z
	err = np.absolute(err)
	plt.hist(err,bins='auto')
	plt.show()
	meanerr= np.mean(err)
	return [meanerr,z,w]

def NormalEqnSoln(x,y):
	y=y[:,None]
	w = np.dot(x.T,x)
	w = np.linalg.inv(w)
	w = np.dot(w,x.T)
	w = np.dot(w,y)
	return w

def AddGaussianFeature(x):

	m = np.size(x,axis=0)	
	x = x[:,None]
	var = np.var(x)
	num = ((x-x[1])/var)**2
	num = -1*num/2
	num = np.exp(num)
	return num



def MultiClassLogisticModel(xtrain,ytrain,x,step,xtest,ytest):
	d = np.size(xtrain,axis=1)
	m = np.size(xtrain,axis=0)
	b = np.ones((m,1))
	xtrain = np.hstack((b,xtrain))
	ytrain = oneHot(ytrain,k)

	w=np.random.rand(k,d+1)

	fig = plt.figure()
	i=0
	lprev=10000
	while(True):
		z = np.dot(xtrain,w.transpose())
		a=softmax(z)
		# print a[0:10,:]
		print np.shape(a)
		l=MSELoss(a,ytrain)
		l=np.sum(l)/m
		print l
		if(i>5000):
			break
		plt.scatter(i,l,color='black',s=2)
		g = gradientForSoftmax(xtrain,ytrain,z)
		w=w+step*g
		lprev=l
		i+=1
	name = "LogisticReg_"+str(step)+".png"
	plt.savefig(name)
	plt.clf
	acc = testing(xtest,ytest,w)
	return acc

def visualizingHyperPlane(x,w):
	plt3d = plt.figure().gca(projection='3d')
	color=['red','blue','green','yellow','orange','purple','black','pink','aqua']
	for i in range(0,k):
		plt3d.scatter(x[i][:,0],x[i][:,1],x[i][:,2],color=color[i],s=2)


	xx, yy = np.meshgrid(range(4), range(4))
	plt1 = plt.figure().gca(projection='3d')
	z1 = (-xx*w[0][1]-yy*w[0][2]-w[0][0])/w[0][3]
	plt1.plot_surface(xx,yy,z1, color='red')
	plt3d.plot_surface(xx,yy,z1, color='red')
	z2 = (-xx*w[1][1]-yy*w[1][2]-w[1][0])/w[1][3]
	plt1.plot_surface(xx,yy,z2, color='blue')
	plt3d.plot_surface(xx,yy,z2, color='blue')
	z3 = (-xx*w[2][1]-yy*w[2][2]-w[2][0])/w[2][3]
	plt1.plot_surface(xx,yy,z3, color='green')
	plt3d.plot_surface(xx,yy,z3, color='green')
	Visualiztion3D(x)
	plt.show()
	
def testing(xtest,ytest,w):
	m1 = np.size(xtest,axis=0)
	b =np.ones((m1,1))
	xtest=np.hstack((b,xtest))
	ytest=oneHot(ytest,k)
	z =np.dot(xtest,w.transpose())
	y1 = np.argmax(z,axis=1)
	y2=np.argmax(ytest,axis=1)
	acc = calculateAccuracy(y1,y2)
	return acc

def softmax(z):
	a = np.exp(z)
	s=np.sum(a,axis=1)
	a=a/s[:,None]
	return a

def gradientForSoftmax(x,y,z):
	a = np.exp(z)
	print a[0:10,:]
	s=np.sum(a,axis=1)
	a1=a/s[:,None]
	g = 2*(y-a1)
	g = np.dot(g.T,x)
	return g

def MSELoss(z,y):
	l = np.dot((y-z).transpose(),(y-z))
	i = len(np.shape(y))
	if(i!=1):
		i=np.size(y,axis=1)
	l = np.eye(i)*l
	l = np.sum(l,axis=0)
	return l

def MSEGradient(z,y,x):
	g = 2*np.dot((z-y).transpose(),x)
	return g

def normalize(x):
	mean = np.mean(x,axis=0)
	var=np.var(x,axis=0)
	x=(x-mean)/(var+0.000001)
	return x

def svm(xtrain,ytrain,c):
	m=np.size(ytrain)
	ytrain = np.reshape(ytrain,(m,1))
	k = ytrain*xtrain
	k = np.dot(k,k.transpose())
	a = ytrain.transpose()
	q = np.ones((m,1))*-1
	g1 = np.eye(m)*-1
	g2 = np.eye(m)
	g = np.vstack((g1,g2))
	h1= np.zeros((m,1))
	h2 = np.ones((m,1))*c
	h=np.vstack((h1,h2))
	P = cvxopt.matrix(k,tc='d')
	A = cvxopt.matrix(ytrain.reshape(1,-1),tc='d')
	q = cvxopt.matrix(q,tc='d')
	G = cvxopt.matrix(g,tc='d')
	h = cvxopt.matrix(h,tc='d')
	b = cvxopt.matrix(np.zeros(1),tc='d')
	sol = cvxopt.solvers.qp(P, q, G, h, A, b)
	mew = np.array(sol['x'])
	w = np.sum(ytrain*mew*xtrain,axis=0)
	w=w[:,None]
	S = (mew > 1e-4).flatten()
	b = ytrain[S] - np.dot(xtrain[S],w)
	b = b[0]
	return [w,b]

def svmOneVsAll(xtrain,ytrain,xtest,ytest,c):
	Ytrain=sepDataForOneVsAll(ytrain,k)
	m=np.size(xtrain,axis=0)
	Z=np.zeros((k,m))
	for i in range(0,k):
		y = Ytrain[i]
		y= labelForSVM(y)
		w,b=svm(xtrain,y,c)
		print w
		z = np.dot(xtrain,w)+b
		Z[i][:,None]=z
	Z=Z.transpose()
	print np.shape(Z)
	print Z[:,0:10]
	ycalc=np.argmax(Z,axis=1)
	print calculateAccuracy(ycalc,ytrain)

def sepDataForOneVsAll(y,k):	
	Y=[]
	print np.shape(y)
	for i in range(0,k):
		temp = np.copy(y)
		ind = np.where(temp[:]!=i)
		temp[ind]=-1
		ind = np.where(temp[:]!=-1)
		temp[ind]=1
		ind = np.where(temp[:]==-1)
		temp[ind]=0
		print np.sum(temp)
		Y.append(temp)
	return Y

def labelForSVM(y):
	ind=np.where(y[:]==0)
	y[ind]=-1
	return y #Converts 0,1 labels to -1,1 labels 

def MedicalData():
# For Medical Data

	dtrain = genfromtxt('medicalData.txt')
	dtest = genfromtxt('medicaltest.txt')
	k = 3 # Number of classes
	N = np.size(dtrain,axis=0)
	xtrain=dtrain[:,1:]
	ytrain=dtrain[:,0]
	xtest = dtest[:,1:]
	ytest = dtest[:,0]

	x = separateByclass(xtrain,ytrain,k)

	# A=[]
	# for i in range (1,100):
	# 	print i
	# 	step = 0.001*i
	# acc = MultiClassLogisticModel(xtrain,ytrain,x,0.01,xtest,ytest)
	# 	print acc
	# 	A.append(acc)

	# fig = plt.figure()
	# plt.scatter(np.arange(100)*0.001,A,color='black',s=3)

	# plt.savefig('LinearRegressionVariationWithStepSize.png')

	# svmOneVsAll(xtrain,ytrain,xtest,ytest,9)
	perceptronOneVsAll(xtrain,ytrain,xtest,ytest)

def RiverData():
	#For River Data

	d = genfromtxt('river.txt')
	m = np.size(d,axis=0)
	dtrain = d[0:2*m/3,:]
	dtest = d[2*m/3:,:] 
	N = np.size(dtrain,axis=0)
	xtrain=dtrain[:,0]
	ytrain=dtrain[:,1]
	xtest = dtest[:,0]
	ytest = dtest[:,1]
	x = np.copy(xtrain)
	x1 = np.copy(xtest)
	polynomial = 5
	# gf = AddGaussianFeature(x)
	xtrain1 =  addPolynomialFeatures(x,polynomial)
	xtest1 = addPolynomialFeatures(x1,polynomial)
	acc , z , w= LinearRegModel(xtrain1,ytrain,xtest1,ytest)
	print "Mean error is ",acc
	plt.scatter(np.arange(np.size(ytest,axis=0)),ytest,color='red',marker='X',s=1)
	plt.scatter(np.arange(np.size(z,axis=0)),z,color='green',marker='o',s=1)
	plt.savefig("PredictedVsActualRiverDataTest.png")
	plt.clf()
	xplot = np.arange(0,120,1)
	xplot = addPolynomialFeatures(xplot,polynomial)
	bplot = np.ones((120,1))
	xplot = np.hstack((bplot,xplot))
	yplot = np.dot(xplot,w)
	plt.scatter(xtest,ytest,color='red',s=1)
	plt.scatter(np.arange(0,120,1),yplot,color='green',s=5)
	plt.show()


# k=3
# MedicalData()
RiverData()