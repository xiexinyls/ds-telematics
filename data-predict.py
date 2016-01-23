
import numpy as np

import collections as clt

import scipy.stats as st
import scipy.spatial.distance as dist

from scipy import linalg

import pandas as pd
import os
import json

from sklearn import cluster
from sklearn.decomposition import PCA
from sklearn import linear_model
from sklearn import svm
from sklearn import tree


np.set_printoptions(precision=4)

def readData(featurepath):
   f = open(featurepath, 'r')
   featuredict = json.load(f)

   return featuredict



def dimReduce( feature ):
   pca = PCA(n_components=5)
   pca.fit( feature )
   outfeature = pca.transform( feature )

   return outfeature



def anomalyDetect( feature ):
   ndata  = feature.shape[0]
   nfeature = feature.shape[1]
#------------------------------------------------------------------------
#Low-Dimensional Method
#------------------------------------------------------------------------

#------------------------------------------------------------------------
#simple anomaly detection
#------------------------------------------------------------------------
   #vmeanavg = np.mean(feature[:,0] )
   #vmeanstd = np.std(feature[:,0] )
   #vstdavg  = np.mean(feature[:,1])
   #vstdstd  = np.std(feature[:,1])
   #gauskernvmean = st.norm.pdf(feature[:,0], loc=vmeanavg, scale=vstdstd )
   #gauskernvstd = st.norm.pdf(feature[:,1], loc=vstdavg, scale=vstdstd )
   #label = np.where( np.logical_or((gauskernvmean>=0.02), (gauskernvstd>=0.02)), 1, 0)

#------------------------------------------------------------------------
#Multivariate Gaussian
#------------------------------------------------------------------------
   featurecov = np.cov( feature.T )
   featuremean = np.mean( feature, axis=0)
   invfeaturecov = linalg.inv( featurecov )

   mdist = np.empty( ndata )
   for i in range(ndata):
      mdist[i] = dist.mahalanobis( feature[i,:], featuremean, invfeaturecov)

   chi2 = st.chi2( nfeature )
   label = np.where( mdist <= chi2.ppf(0.4), 1, 0)

   #pltx1d = np.linspace( np.min( feature[:,0])-1, np.max( feature[:,0])+1, 50 )
   #plty1d = np.linspace( np.min( feature[:,1])-1, np.max( feature[:,1])+1, 80 )
   #[pltx, plty] = np.meshgrid( pltx1d, plty1d)
   #pos = np.empty(pltx.shape + (3,))
   #pos[:,:,0] = pltx
   #pos[:,:,1] = plty
   #pos[:,:,2] = featuremean[2]*np.ones( pltx.shape )
   #rv = st.multivariate_normal( mean=featuremean, cov=featurecov)
   #pdf = rv.pdf( pos )
   #plt.contourf( pltx, plty, pdf )
   #plt.plot( feature[:,0], feature[:,1], 'ko' )
   #plt.colorbar()
   #plt.xlabel('vmeandang')
   #plt.ylabel('vskew')
   #plt.show()

##3D Plot of the Adv Features
      #fig = plt.figure()
      #ax = Axes3D(fig)
      #ax.scatter( feature[:,0], feature[:,1], feature[:,2] )
      #ax.set_xlabel('V Mean')
      #ax.set_ylabel('V Variance')
      #ax.set_zlabel('V Skewness')
      #plt.show()

#------------------------------------------------------------------------
#Distance-based anomaly detection
#------------------------------------------------------------------------
      #distperc = np.empty(ndata)

      #featurecov = np.cov( feature.T )
      #featuremean = np.mean( feature, axis=0)
      #invfeaturecov = linalg.inv( featurecov )
      #mdist = np.empty( ndata )
      #for j in range(ndata):
         #mdist[j] = dist.mahalanobis( feature[j,:], featuremean, invfeaturecov)
      #distrange = np.median( mdist )

      #for i in range(ndata):
         #for j in range(ndata):
            #mdist[j] = dist.mahalanobis( feature[i,:], feature[j,:], invfeaturecov)
         #distperc[i] = 1.*np.count_nonzero( mdist < distrange )/ndata
      #label = np.where( distperc < 0.05 , 0, 1)


#------------------------------------------------------------------------
#kmean method
#------------------------------------------------------------------------
   #k_means = cluster.KMeans(n_clusters=3)
   #k_means.fit(feature)
   #labels = k_means.labels_
   #for i in range(3):
      #labelind = (labels == i)
      #plt.plot( feature[labelind,0], feature[labelind,1], lw=1 )

   #plt.savefig('../plot/'+strdriver+'.png', dpi=100)
   #plt.clf()
   #plt.show()
#------------------------------------------------------------------------
#High-Dimensional Method
#------------------------------------------------------------------------

#------------------------------------------------------------------------
#Correlation-based anomaly detection
#------------------------------------------------------------------------
   #distperc = np.empty(ndata)
   #corrcoef = np.abs( np.corrcoef( feature ) )
   #for i in range(ndata):
      #distperc[i] = 1.*np.count_nonzero( corrcoef[i,:] > 0.8 )/ndata
   #label = np.where( distperc < 0.5 , 0, 1)


#------------------------------------------------------------------------
#Cosine-based anomaly detection
#------------------------------------------------------------------------
   #width = np.empty(ndata)
   #cosdist = np.empty( ndata )
   #for i in range(ndata):
      #for j in range(ndata):
         #cosdist[j] = dist.cosine( feature[i], feature[j])
      #width[i] = np.std(cosdist)
   #label = np.where( width < 0.12 , 0, 1)


#------------------------------------------------------------------------
#Grid-based anomaly detection
#------------------------------------------------------------------------
   #nbin = 3
   #xgrid = np.empty( (ndata, nfeature) )
   #featuremin = np.min( feature, axis=0)
   #featuremax = np.max( feature, axis=0)
   #for i in range(nfeature):
      #featurebin = np.linspace( featuremin[i], featuremax[i], nbin+1 )
      #for k in range(nbin):
         #featureind = np.logical_and( (feature[:,i] >= featurebin[k]), \
               #(feature[:,i] <= featurebin[k+1]) )
         #xgrid[featureind, i] = k
   #xgridtuple = map(tuple, xgrid)
   #xgriddict = clt.Counter( xgridtuple )
   #print( xgriddict )

   return label


clf = linear_model.LogisticRegression(C=10, penalty='l1')
#clf = svm.SVC()
#clf = linear_model.LassoLars(alpha=.1)
#clf = tree.DecisionTreeClassifier()
#clf = linear_model.LinearRegression()
#outliers_fraction = 0.25
#outly = svm.OneClassSVM(nu=0.95*outliers_fraction+0.05, kernel="rbf", gamma=0.1)

if __name__ == "__main__":

   f = open('list', 'r')
   listfstr = np.array( f.read().splitlines() )
   nlistfstr = len( listfstr )

   featurename = 'feature3'

   datapath='./'+featurename
   featuredict = readData( datapath )
   ndriver = len(featuredict)
   trainind = np.arange(ndriver)

   print('driver_trip,prob')
   fout = open('output', 'w')
   fout.write('driver_trip,prob\n')


   for i in range(nlistfstr):
   #for i in [10]:
      strdriver = listfstr[i]
      traindata1 = np.array(featuredict[strdriver] )
      ntraindata1 = len( traindata1 )
      nfeature = len( traindata1[0] )
      print('feature number:',nfeature)


      cvsize = 20
      nfold = ntraindata1/cvsize

      label = np.empty( ntraindata1 )
      nfdriver = 10

      if ( (i+nfdriver) < nlistfstr ):
         drv0ind = range( i+1, i+1+nfdriver)
      else:
         drv0ind = range( i-1, i-1-nfdriver, -1)

#prepare drvier 0
      traindata0 = []
      traindata0.append( np.array( featuredict[ listfstr[ drv0ind[0] ] ] ) )
      for i in range(1, nfdriver):
         traindata0.append( np.array( featuredict[ listfstr[ drv0ind[i] ] ] ) )
      traindata0 = np.array( traindata0)




      ##for ifold in range(nfold):
      #for ifold in [0, 1, 2]:
         #begind = ifold*cvsize
         #endind = (ifold+1)*cvsize
         #cvind = np.empty( ntraindata1, dtype=bool )
         #cvind[:] = True
         #cvind[begind:endind] = False
         #testind = np.logical_not( cvind )
         #ncv = len(cvind)

##prepare driver 1
         #traindata1cv = np.tile( traindata1[cvind], (nfdriver,1) )
##prepare driver 0
         #traindata0cv = np.reshape( traindata0[:,cvind,:], (-1,3) )


##get final train feature
         #traindatacv = np.append( traindata1cv, traindata0cv, axis=0)

##assign final train label
         #ntraindata0cv = len(traindata0cv)
         #ntraindata1cv = len(traindata1cv)
         #trainlabelcv = np.empty( ntraindata1cv+ntraindata0cv )
         #trainlabelcv[0:ntraindata1cv] = 1
         #trainlabelcv[ntraindata0cv:] = 0

###train model
         ##selfeatureind = [0, 1, 2]
         ##clf.fit(traindata[:,selfeatureind], trainlabel) 
         #clf.fit(traindatacv, trainlabelcv) 

         #testdata1cv = traindata1[testind]
         ##model = clf.predict( testdata[:,selfeatureind] ).astype(int)
         #testlabelcv = clf.predict( testdata1cv ).astype(int)
         #label[testind]  = testlabelcv.astype(int)


#prepare driver 1
      traindata1cv = np.tile( traindata1, (nfdriver,1) )
#prepare driver 0
      traindata0cv = np.reshape( traindata0, (-1,nfeature) )


#get final train feature
      traindatacv = np.append( traindata1cv, traindata0cv, axis=0)

#assign final train label
      ntraindata0cv = len(traindata0cv)
      ntraindata1cv = len(traindata1cv)
      trainlabelcv = np.empty( ntraindata1cv+ntraindata0cv )
      trainlabelcv[0:ntraindata1cv] = 1
      trainlabelcv[ntraindata0cv:] = 0

##train model
      selfeatureind = [0, 1, 2]
      clf.fit(traindatacv[:,selfeatureind], trainlabelcv) 
      #clf.fit(traindatacv, trainlabelcv) 

      label = clf.predict( traindata1[:,selfeatureind] ).astype(int)
      #label = clf.predict( traindata1 ).astype(int)




   ##for i in range(nlistfstr):
      ##strdriver = listfstr[i]
      ##traindata = np.array(featuredict[strdriver] )

##scikit-learn oneclass SVM
      ##outly.fit( traindata )
      ##y_pred = outly.decision_function(traindata).ravel()
      ##threshold = st.scoreatpercentile(y_pred,100 * outliers_fraction)
      ##y_pred = y_pred > threshold
      ##label = np.where( y_pred, 1, 0)
##my anomaly detection
      ##label = anomalyDetect( traindata )



      print( strdriver )
#output to file 'output'
      for i in range(len(label)):
         fout.write(strdriver+'_'+str(i+1)+','+str(label[i])+'\n')
         print(strdriver+'_'+str(i+1)+','+str(label[i]) )


