
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import scipy.stats as st
import scipy.spatial.distance as dist

from scipy import linalg

import pandas as pd
import csv
import os

from sklearn import cluster

#from colorline import colorline


np.set_printoptions(precision=4)

def readData(datapath, strdriver, itrip):
   '''
   (int, int) -> float array
   return the pandas DataFrame from #idriver #itrip
   '''
   #with open('../mlproject-data/drivers/'+str(dnum)+'/1.csv', 'r') as f:
       #reader = csv.reader(f)
       #for row in reader:
           #print row

   #f = open('../mlproject-data/drivers/'\
         #+str(idriver)+'/'+str(itrip)+'.csv', 'r')
   #x = line.split(',') for line in f

   xy = pd.read_csv(datapath+'/'\
         +str(strdriver)+'/'+str(itrip)+'.csv')
   xy = np.array(xy)

   return xy[:,0], xy[:,1]



def getBasicFeature( x, y ):
   '''
   (x,y) array -> velocity, acceleration, angle, diff angle
   '''
   dx = np.diff( x )
   dy = np.diff( y )
#convert to velocity v and accelaeration dv
   v = 2.23*np.sqrt(dx**2+dy**2)
   dv = np.diff( v )
   v = (v[1:]+v[:-1])/2.
   #plt.plot(v)
   #plt.show()

   #v = (v[1:]+v[:-1])/2.
   #for i in range( len(v) ):
      #print( v[i], dv[i] )
#convert to angles and delta angles
   ang = np.arctan2(dy,dx)/np.pi*180
   dang = np.diff( ang )
   ang = (ang[1:]+ang[:-1])/2.

   return [ v, dv, ang, dang ]



def getAdvFeature( v, dv, ang, dang):
   '''
   (v, dv, ang, dang) array -> advanced features
   '''
   v = np.where( v > 90, np.mean(v), v)
   vmeandang = np.abs(v*dang)
   vmeandang = np.mean(vmeandang)
#simple adv features
   vmean = np.mean(v)
   vstd  = np.std(v)
   vskew = st.skew(v)

   ap75 = np.percentile( dv[dv>0], 75)
   an25 = np.percentile( dv[dv<0], 25)

   #return [ vmean, vstd, vskew ]
   #return [ vmean, vskew, an25 ]

#high dimensional features
#for speed
   #vbins = np.linspace(5,75,36)
   #vhist = np.histogram( v, bins=vbins, density=True )
   #plt.hist( v, bins=vbins, normed=True )
   #plt.xlabel('Speed mile/hour')
   #plt.show()
   #return vhist[0]

#for acceleration
   #acbins = np.linspace(0,5,50)
   #dvhist = np.histogram( dv[dv>0], bins=acbins, density=True )
   #plt.hist( dv[dv>0], bins=acbins, normed=True )
   #plt.xlabel('Acceleration mile/hour/s')
   #plt.show()
#for deceleration
   dcbins = np.linspace(-5,0,50)
   dvhist = np.histogram( dv[dv<0], bins=dcbins, density=True )
   #plt.hist( dv[dv<0], bins=dcbins, normed=True )
   #plt.xlabel('Deceleration mile/hour/s')
   #plt.show()
   if np.any( np.isnan( dvhist[0] ) ):
      dvhist[0][:] = 0.1
   return dvhist[0]



def getDriver( strdriver ):
   ntrip = len([name for name in os.listdir(datapath+'/'+strdriver)])
   outfeatures = []
   for i in range(ntrip):
   #for i in [1]:
      x, y = readData(datapath, strdriver, i+1)
      [ v, dv, ang, dang ] = getBasicFeature(x, y)

      feature = getAdvFeature(v, dv, ang, dang)
      outfeatures.append(feature)
   return np.array( outfeatures )



if __name__ == "__main__":

   datapath='../mlproject-data/drivers'
   f = open('list', 'r')
   listf = f.read().splitlines()

   print('driver_trip,prob')
   fout = open('output', 'w')
   fout.write('driver_trip,prob\n')

   for strdriver in listf:
   #for strdriver in ['10']:
      print(strdriver)
      feature = getDriver(strdriver)
      ndata  = feature.shape[0]
      nfeature = feature.shape[1]


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
#Correlation-based anomaly detection
#------------------------------------------------------------------------
      distperc = np.empty(ndata)

      corrcoef = np.abs( np.corrcoef( feature ) )
      for i in range(ndata):
         distperc[i] = 1.*np.count_nonzero( corrcoef[i,:] > 0.2 )/ndata
      label = np.where( distperc < 0.5 , 0, 1)



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
#Multivariate Gaussian
#------------------------------------------------------------------------
      #featurecov = np.cov( feature.T )
      #featuremean = np.mean( feature, axis=0)
      #print(feature)
      #print(featuremean)
      #invfeaturecov = linalg.inv( featurecov )

      #mdist = np.empty( ndata )
      #for i in range(ndata):
         #mdist[i] = dist.mahalanobis( feature[i,:], featuremean, invfeaturecov)

      #chi2 = st.chi2( nfeature )
      #label = np.where( mdist <= chi2.ppf(0.7), 1, 0)


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
#simple anomaly detection
#------------------------------------------------------------------------
      #vmeanavg = np.mean(feature[:,0] )
      #vmeanstd = np.std(feature[:,0] )
      #vstdavg  = np.mean(feature[:,1])
      #vstdstd  = np.std(feature[:,1])

      #gauskernvmean = st.norm.pdf(feature[:,0], loc=vmeanavg, scale=vstdstd )
      #gauskernvstd = st.norm.pdf(feature[:,1], loc=vstdavg, scale=vstdstd )

      #label = np.where( np.logical_or((gauskernvmean>=0.02), (gauskernvstd>=0.02)), 1, 0)


      for i in range(len(label)):
         fout.write(strdriver+'_'+str(i+1)+','+str(label[i])+'\n')
         print(strdriver+'_'+str(i+1)+','+str(label[i]) )


