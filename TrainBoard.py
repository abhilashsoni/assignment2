import numpy as np 
import matplotlib.pyplot as plt
from numpy import genfromtxt
from numpy import linalg as LA


def dataProcessing(d,k):
        N = np.size(d,axis=0)
        dtrain = d[0:int(5*N/6),:]
        dtest = d[int(5*N/6):,:]
        xtrain=dtrain[:,1:]
        ytrain=dtrain[:,0]
        xtest = dtest[:,1:]
        ytest = dtest[:,0]
        x = []
        for i in range(0,k):
                t = dtrain[dtrain[:,0]==i,:]
                x.append(t[:,1:])

        return [xtrain,ytrain,xtest,ytest,x]


def separateByclass(x,y,k):
	X= []
	for i in range(0,k):
		t = x[y[:]==i,:]
		X.append(t)
	return X


	
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

def MLE(x):
	mean = np.mean(x,axis=0)
	var = np.cov(x.transpose())
	print np.size(mean)
	return [mean,var]

def MLENaiive(x):
	mean = np.mean(x,axis=0)
	var = np.var(x,axis=0)
	return [mean,var]


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
	err = np.where(err==0)
	acc = np.size(err)*1.0/np.size(ycalc)*100
	statistics(ycalc,ytest,k)
	return acc

def calculatePrecision(ycalc,ytest):
	deno = 0
	nomi = 0
	for j in range(0,k-1):
		err = np.where(ycalc==i and ytest==i)
		err1 = np.where(ycalc==i and ytest!=i)
		nomi = nomi + np.size(err)
		deno = deno + np.size(err1)
	return nomi/(deno+nomi);

def calculateRecall(ycalc,ytest):
	deno = 0
	nomi = 0
	for j in range(0,k-1):
		err = np.where(ycalc==i and ytest==i)
		err1 = np.where(ycalc!=i and ytest==i)
		nomi = nomi + np.size(err)
		deno = deno + np.size(err1)
	return nomi/(deno+nomi);

def Naiivegaussian(x,mu,var):
	d = np.size(x,axis=1)
	n = np.size(x,axis=0)
	sig = np.eye((d))
	sigma = sig*var
	sigmainv=sig/var
	det = np.product(np.sum(sigma,axis=0))
	z = np.dot(x-mu,sigmainv)
	z = np.dot(z,(x-mu).transpose())
	z = z * np.eye(n)
	z = np.sum(z,axis=0)
	z = np.exp(-1*z)
	z = z/(((2*3.14)**(d/2))*(det**(1/2)))
	return z

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
	m = m.transpose()
	print "Confusion Matrix"
	print m
	totpredicted = np.sum(m,axis=1)
	totactual = np.sum(m,axis=0)
	p = np.eye((k))*m
	p = np.sum(p,axis=0)*1.0
	precision = np.divide(p,totpredicted)
	recall = np.divide(p,totactual)
	f1=2*precision*recall
	f2=precision+recall
	fscore = f1/f2
	print "Precision "
	print  precision
	print "Recall "
	print recall
	print "Fscore"
	print fscore


def classifyBayes(x,xtest,ytest,N):
	theta = []
	cc = []
	prior = []
	posterior = []
	for i in range(0,k):
		t = MLE(x[i])
		ccond = gaussian(xtest,t[0],t[1])
		p = np.size(x[i],axis=0)*1.0/N
		pos = ccond*p
		theta.append(t)
		cc.append(ccond)
		prior.append(p)
		posterior.append(pos)
	bayes = np.argmax(posterior,axis=0)
	acc = calculateAccuracy(bayes,ytest)
	print "Accuracy on test set for bayes classifier is ",acc
	return acc


def classifyNaiiveBayes(x,xtest,ytest,N):
	theta = []
	cc = []
	prior = []
	posterior = []
	for i in range(0,k):
		t = MLENaiive(x[i])
		ccond = Naiivegaussian(xtest,t[0],t[1])
		p = np.size(x[i],axis=0)*1.0/N
		pos = ccond*p
		theta.append(t)
		cc.append(ccond)
		prior.append(p)
		posterior.append(pos)

	bayes = np.argmax(posterior,axis=0)
	acc = calculateAccuracy(bayes,ytest)
	print "Accuracy on test set for naiive bayes classifier is ",acc
	return acc



def classifyKNN(xtrain,ytrain,xtest,ytest, knn):
	n = np.size(xtest,axis=0)
	res = distanceMetric(xtrain,xtest)
	order = np.argsort(res,axis=1)
	order = order[:,0:knn]
	y = np.reshape(ytrain,(np.size(ytrain),1))
	res=y[order]
	res=np.squeeze(res)
	ans=np.zeros((k,np.size(xtest,axis=0)))
	for i in range(0,k):
		f=np.sum((res==i)*1,axis=1)
		ans[i]=f

	ans=ans.transpose()
	ans=np.argmax(ans,axis=1)
	acc=calculateAccuracy(ans,ytest)
	print "Accuracy for k-nearest neighbour with k = ",knn," is ",acc
	return acc

def kMeans(xtrain,km):
	theta = MLENaiive(xtrain)
	var = theta[1]
	mu = theta[0]
	rnd = np.random.rand(km,np.size(xtrain,axis=1))
	centroid = var*rnd+mu
	
	c=0
	while(True):

		prev=np.copy(centroid)
		dist = distanceMetric(xtrain,centroid)
		curr = np.argmin(dist,axis=0)
		for i in range(0,km):
			temp=np.where(curr==i)
			x1=xtrain[temp]
			centroid[i]=np.mean(x1,axis=0)
		update=distanceMetric(prev,centroid)
		update=update*np.eye(km,km)
		update=np.sum(update,axis=0)
		mindiff=np.min(update)
		kMeansVisualisation(xtrain,centroid,km,c)
		c+=1
		print c
		print mindiff
		if(mindiff<=0.00000001):
			break;


	plt.clf()
	plt.scatter(xtrain[:,0],xtrain[:,1],color='grey',s=2)
	plt.savefig('train_unclustered.png')



	# print centroid


def kMeansVisualisation(xtrain,centroid,km,i1):
	plt.figure(1)
	dist = distanceMetric(xtrain,centroid)
	curr = np.argmin(dist,axis=0)
	color=['red','blue','green','yellow','orange','purple','black','pink','aqua']
	for i in range(0,km):
		temp=np.where(curr==i)
		x1=xtrain[temp]
		plt.scatter(centroid[i][0],centroid[i][1],s=100,color='black',marker='X')
		plt.scatter(x1[:,0],x1[:,1],s=2,color=color[i])
	name=str(i1)+'.png'
	print name
	plt.savefig(name)
	plt.clf()


def OptimumKNN(xtrain,ytrain,xtest,ytest):
	res=[]
	index=[]
	for i in range(k,100):
		t=classifyKNN(xtrain,ytrain,xtest,ytest,i)
		res.append(t)
		index.append(i)

	opt = np.argmax(res)
	print "Optimum K-nearest neighbour is at K=",opt+3," and gives accuracy ",res[opt]

	plt.figure(1)
	plt.plot(index,res)
	plt.show()



dtrain = genfromtxt('TrainData.txt')
#dtest = genfromtxt('trainTest.txt')
k = 2 # Number of classes
N = np.size(dtrain,axis=0)
#xtrain=dtrain[:,1:]
#ytrain=dtrain[:,0]

dp = dataProcessing(dtrain,k)
xtrain = dp[0]
ytrain = dp[1]
xtest = dp[2]
ytest = dp[3]
x = dp[4]
N = np.size(xtrain,axis=0)
#xtest = dtest[:,1:]
#ytest = dtest[:,0]
#x = separateByclass(xtrain,ytrain,k)
#classifyNaiiveBayes(x,xtest,ytest,N)
classifyBayes(x,xtest,ytest,N)
#kMeans(xtrain,k)



#OptimumKNN(xtrain,ytrain,xtest,ytest)
# xtrain1 = PCA(xtrain,2)
# xtest1 = PCA(xtest,2)
# x1 = separateByclass(xtrain1,ytrain,3)
# classifyNaiiveBayes(x1,xtest1,ytest,N)
# OptimumKNN(xtrain1,ytrain,xtest1,ytest)
# DataVisualization(x1)

