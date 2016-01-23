# Driver Telematics Analysis from Kaggle

AXA has raised a competition in www.kaggle.com to identify trips which are not from the driver of interest. The dataset AXA provides over 50,000 anonymized driver trips, expressed as instant
coordinate, that is, a pair of (x,y) coordinate per second, from which we can extract useful features, such as velocity and acceleration. Each driver is associated with 200 trips, including
both real trips as well as a small and random number of false trips (trips that were not driven by the driver of interest). At a first glance, this is a unsupervised problem because trip labels are
not given. This is actually a anomaly detection problem because the final mission is to identify the false or anomaly trips.

Here is my code to solve this driver fingerprint extraction problem.

plot/ is blank directory used to save trip data visualization figures
drivers/ is a test dataset. Sub-directory is the driver number. Under drivers/1/, there are 200 cvs trip files for driver number one.

data-prep.py is a Python script used to do feature extraction and trip data visualization. The feature data will be saved to a JSON file "dictfeature".

data-predict.py is a Python script used to determine whether the 200 trip data under each driver directory is anomaly or not with many anomaly detection methods.

