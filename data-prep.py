

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D

import scipy.stats as st

import pandas as pd

import os
import json


np.set_printoptions(precision=4)
plotpath = './plot'

#parameter setting for high dimensional feature extraction
vbins = np.linspace(5,75,21)
dvbins = np.linspace(-5,5,21)
danglebins = np.linspace(-5,5,21)

def readData( tripfilename ):
    '''
    return the pandas DataFrame from #tripfilename
    '''
    xy = pd.read_csv( tripfilename )
    xy = np.array(xy)

    return xy



def procData( xy ):
   '''
   turn (x,y) data into meaningful information
   (x,y) list -> velocity, acceleration, angle, diff angle
   '''
   x = xy[:,0]
   y = xy[:,1]
   dx = np.diff( x )
   dy = np.diff( y )
#convert to velocity v in MPH
   v = 2.23*np.sqrt(dx**2+dy**2)
   v = np.where( v > 90, np.mean(v), v)
#and accelaeration 
   dv = np.diff( v )

#let the v dim equal to acceleration dim
   v = (v[1:]+v[:-1])/2.

#convert to angles and delta angles
   ang = np.arctan2(dy,dx)/np.pi*180

   dang = np.empty( len(v) )
   for i in range(len(v)):
      if ( np.abs( ang[i+1]-ang[i] ) > 180 ):
         if ( ang[i] < 0):
            dang[i] = ang[i+1]-360-ang[i]
         else:
            dang[i] = ang[i+1]+360-ang[i]
      else:
         dang[i] = ang[i+1]-ang[i]
   dang = np.where( np.abs(dang) < 90, dang, 0 )
   ang = (ang[1:]+ang[:-1])/2.

   return [ v, dv, ang, dang ]



def getFeature( v, dv, angle, dangle):
   '''
   (v, dv, angle, dangle) array -> advanced features
   '''
#could do some denoising
   #v = savitzky_golay(v, 5, 3)
   #ang = savitzky_golay(ang, 11, 3)

   meanvdangle = np.mean( np.abs(v*dangle) )
   absdangle = np.mean( np.abs( dangle ) )

#simple low dim feature statistics
   vmean = np.mean(v)
   vstd  = np.std(v)
   vskew = st.skew(v)
#accelaeration mean
   apind = dv>0
   if ( np.sum(apind) == 0):
      ap = 0
   else:
      ap = np.mean( dv[apind] )
#breaking mean
   anind = dv<0
   if ( np.sum(anind) == 0):
      an = 0
   else:
      an = np.mean( dv[anind] )

   if ( len(v) < 5 ):
      vmean = 0
      vstd = 0
      vskew = 0
      ap = 0
      an = 0
      absdangle = 0

# high dim feature extraction
#for speed
   vhist, vbin = np.histogram( v, bins=vbins, density=True )
   if np.any( np.isnan( vhist ) ):
      vhist[:] = 0

#for +/- acceleration
   dvhist, dvbin = np.histogram( dv, bins=dvbins, density=True )
   if np.any( np.isnan( dvhist ) ):
      dvhist[:] = 0

#choose what features to output
#output low dim
   return [ vmean, vstd, vskew ]
   #return [ vmean, vstd, vskew, ap75, an25, absdangle, meanvdangle ]
   #return [ vmean, vstd, vskew, ap, an, absdangle ]
#output high dim
   #return np.append(vhist, dvhist)




def plotTrip( xy, v, dv, angle, dangle, strdriverno, tripno):
    """
    a ploting fucntion for trip data visualization
    """
    print("Plotting trip No."+("%5d"%tripno) )

#plot the velocity and acceleration
    plt.gcf().set_size_inches(5,10)
    plt.clf()
    plt.subplot(211)
    plt.plot(v, label='speed')
    plt.plot(dv, label='acceleration')
    plt.legend()
    plt.subplot(212)
    plt.plot(angle, label='angular speed')
    plt.plot(dangle, label='angular acceleration')
    plt.legend()

    plt.tight_layout()
    plt.savefig( plotpath+'/drv'+strdriverno+'-trip'+("%03d"%tripno)+'-v' )


#draw the trip
    cdata = v
    cmin = np.min( cdata )
    cmax = np.max( cdata )
    cdiff = np.max( np.abs( [cmin, cmax] ) )
# let the shading represent velocity
    plt.gcf().set_size_inches(8,8)
    plt.clf()
    segments = np.array( zip( xy[:-1], xy[1:]  ) )
    coll = LineCollection(segments, lw=3, cmap='jet', \
         norm=plt.Normalize(cmin, cmax))
    coll.set_array(cdata)
    plt.gca().add_collection(coll)

    #patches = []
    #for ixy in xy[1:len(xy)-1]:
       #circle = Circle(ixy, np.mean(v)/2 )
       #patches.append(circle)

    #p = PatchCollection(patches, cmap='coolwarm', \
            #norm=plt.Normalize(-10, 10), alpha=0.6)
    #p.set_array(dangle)
    #plt.gca().add_collection(p)

# fit the plot
    xmin = np.min( xy[:,0] )
    xmax = np.max( xy[:,0] )
    diffx = xmax-xmin
    ymin = np.min( xy[:,1] )
    ymax = np.max( xy[:,1] )
    diffy = ymax-ymin
    plt.xlim( xmin-0.1*diffx, xmax+0.1*diffx)
    plt.ylim( ymin-0.1*diffy, ymax+0.1*diffy)

    plt.plot( [0], [0], 'b^', ms=10 )
    plt.colorbar(coll)
    plt.title('driver '+(strdriverno)+' trip '+("%03d"%tripno) )

    plt.tight_layout()
    plt.savefig( plotpath+'/drv'+strdriverno+'-trip'+("%d"%tripno)+'-trip' )



#high dimensional features
    plt.gcf().set_size_inches(12,4)
    plt.clf()
#for speed
    plt.subplot(131)
    plt.hist( v, bins=vbins, normed=True )
    plt.xlabel('Speed mile/hour')
#for +/- acceleration
    plt.subplot(132)
    plt.hist( dv, bins=dvbins, normed=True )
    plt.xlabel('Acceleration mile/hour/s')
#for dang
    plt.subplot(133)
    plt.hist( dangle, bins=danglebins)
    plt.xlabel('Turning Angle')

    plt.tight_layout()
    plt.savefig( plotpath+'/drv'+strdriverno+'-trip'+("%03d"%tripno)+'-hidim' )




def getDriver( drvpath, strdriverno ):
    """
    driverno is a string representing which driver number in the dataset
    """
    trippath = drvpath+'/'+strdriverno
    tripfnlist = os.listdir(trippath)
    ntrip = len(tripfnlist)

    outfeature = []
    for i in range(ntrip):
      xy = readData( trippath+'/'+tripfnlist[i] )
      [ v, dv, ang, dang ] = procData(xy)

#turn of visualization for now.
      #plotTrip( xy, v, dv, ang, dang, strdriverno, i)

      feature = getFeature(v, dv, ang, dang)
      outfeature.append(feature)

    return np.array( outfeature )



class NumPyArangeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist() # or map(int, obj)
        return json.JSONEncoder.default(self, obj)


if __name__ == "__main__":

   datapath='../mlproject-data/drivers'

#get the driver No. list from drvlist file
   with open('drvlist', 'r') as file_drvlist:
       drvlist = []
       for line in file_drvlist:
           drvlist.append( line.strip() )

#Extract features for each driver and
#Save it into a JSON object file for modeling later
   ndrvlist = len( drvlist )
   featuredict = {}
   for i in range(ndrvlist):
   #for i in [0]:
      driverno = drvlist[i]
      print( "Reading Driver "+driverno )
      feature = getDriver(datapath, driverno )
      featuredict[ driverno ] = feature
      
   json.dump(featuredict, open("dictfeature",'w'), cls=NumPyArangeEncoder)



