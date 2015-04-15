
import numpy as np
import matplotlib.pyplot as plt

import scipy.stats as st

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
#convert to angles and delta angles
   ang = np.arctan2(dy,dx)/np.pi*180
   dang = np.diff( ang )

   return [ v, dv, ang, dang ]



def getAdvFeature( v, dv, ang, dang):
   '''
   (v, dv, ang, dang) array -> advanced features
   '''
   vmean = np.mean(v)
   vstd  = np.std(v)
   vskew = st.skew(v)
   vl2   = 0

   #plt.plot(x, gauss)
   #plt.bar( vhist[1][1:], vhist[0] )
   #plt.show()

#simple features
   return [ vmean, vstd, vskew, vl2 ]

#complex features
   #vhist = np.histogram( v, bins=np.linspace(5,75,36), density=True )
   #return vhist


def getDriver( strdriver ):
   ntrip = len([name for name in os.listdir(datapath+'/'+strdriver)])
   outfeatures = []
   for i in range(ntrip):
      x, y = readData(datapath, strdriver, i+1)
      [ v, dv, ang, dang ] = getBasicFeature(x, y)

      [ vmean, vstd, vskew, vl2 ] = getAdvFeature(v, dv, ang, dang)
      outfeatures.append([vmean, vstd, vskew])
   return np.array( outfeatures )


      #vhist = getAdvFeature(v, dv, ang, dang)
      #outfeatures.append( vhist[0] )
   #return np.array(outfeatures)


if __name__ == "__main__":

   datapath='../mlproject-data/drivers'
   f = open('list', 'r')
   listf = f.read().splitlines()

   print('driver_trip,prob')
   fout = open('output', 'w')

   #for strdriver in listf:
   for strdriver in ['10']:
      print(strdriver)
      feature = getDriver(strdriver)
      ndata = feature.shape[0]
      print(feature.shape)

      #pltx1d = np.linspace( np.min( feature[:,0]), np.max( feature[:,0]) )
      #plty1d = np.linspace( np.min( feature[:,1]), np.max( feature[:,1]) )
      #[pltx, plty] = np.meshgrid( pltx1d, plty1d)
      #pltz = 0.*pltx
      ##zz = np.array([pltx, plty, pltz])
      #print( pltx.shape )
      #zz = np.hstack([pltx, plty, pltz])
      #print( zz.shape )


#kmean method
      #k_means = cluster.KMeans(n_clusters=3)
      #k_means.fit(feature) 
      #labels = k_means.labels_
      #for i in range(3):
         #labelind = (labels == i)
         #plt.plot( feature[labelind,0], feature[labelind,1], lw=1 )

      #plt.savefig('../plot/'+strdriver+'.png', dpi=100)
      #plt.clf()
      #plt.show()

#Multivariate Gaussian
      
      featurecov = np.cov( feature.T )
      featuremean = np.mean( feature, axis=0)
      rv = st.multivariate_normal( mean=featuremean, cov=featurecov )
      featurey = rv.pdf( feature )
      chi2 = st.chi2( 3 )
      label = np.where( featurey <= chi2.pdf(0.20), 1, 0)
      #print(label)
      logfeaturey = np.log( featurey )
      plt.hist( logfeaturey )
      plt.show()


      #plt.plot( feature[labelin




#simple anomaly detection
      #vmeanavg = np.mean(feature[:,0] )
      #vmeanstd = np.std(feature[:,0] )
      #vstdavg  = np.mean(feature[:,1])
      #vstdstd  = np.std(feature[:,1])

      #gauskernvmean = st.norm.pdf(feature[:,0], loc=vmeanavg, scale=vstdstd )
      #gauskernvstd = st.norm.pdf(feature[:,1], loc=vstdavg, scale=vstdstd )

      #label = np.where( np.logical_or((gauskernvmean>=0.02), (gauskernvstd>=0.02)), 1, 0)


      #for i in range(len(label)):
         #fout.write(strdriver+'_'+str(i+1)+','+str(label[i])+'\n')
         #print(strdriver+'_'+str(i+1)+','+str(label[i]) )


