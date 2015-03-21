
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv


np.set_printoptions(precision=4)

DATAPATH='../mlproject-data/drivers'

def readData(idriver, itrip):
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

   xy = pd.read_csv(DATAPATH+'/'\
         +str(idriver)+'/'+str(itrip)+'.csv')

   return np.array(xy)



def getFeature( xy ):
   '''
   (x,y) array -> a lot of features
   '''
   x = xy[:,0]
   y = xy[:,1]
   dx = x[1:]-x[:(len(x)-1)]
   dy = y[1:]-y[:(len(y)-1)]
#convert to velocity v and accelaeration dv
   v = 2.23*np.sqrt(dx**2+dy**2)
   dv = v[1:]-v[:(len(v)-1)]
#convert to angles and delta angles
   ang = np.arctan2(dy,dx)/np.pi*180
   dang = ang[1:]-ang[:(len(ang)-1)]

   return [ v, dv, ang, dang ]



if __name__ == "__main__":

   #for i in range(50):
      #xy = readData(1, i+1)
      #[ v, dv, ang, dang ] = getFeature(xy)

      #ind = np.logical_and( (np.abs(dang) > 10), (np.abs(dang) < 30) )
      #print(i)
      ##print(ind)
      #plt.hist( v[ind] )
      #plt.savefig('../plot/'+('%03i' % i)+'.png', dpi=150, bbox_inches='tight', pad_inches=0)
      #plt.clf()
   xy = readData(1,1)
   print(xy)

   #plt.plot(xy[:,0], xy[:,1])
   #plt.show()

   #print(ds)
   #print(dds)
   #print(ang)
   #print(dang)
   #print( len(x), len(dx) )


