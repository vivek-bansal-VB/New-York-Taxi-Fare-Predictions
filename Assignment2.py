
# coding: utf-8

# In[1]:


# used in Data processing
import pandas as pd
import numpy as np
import datetime as dt

# Visualization libraries
import seaborn as sns
import matplotlib.pyplot as plt
from math import sin, cos, sqrt, atan2, radians
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Machine Learning
from sklearn.model_selection import train_test_split

from pandas.tseries.holiday import USFederalHolidayCalendar

# loads data frame
df = pd.read_csv("train.csv",nrows = 50_00_000)


# In[3]:


# check statistics of the given parameters
df.describe()


# In[4]:


df.shape


# In[5]:


# Let's see the first 5 rows of our training data
df.head(5)


# In[6]:


# checking how many rows contain atleast a blank entry in it.
df.isna().sum()


# In[7]:


# data set contains some null entries. Removing them from the dataset
df = df.dropna()


# In[8]:


#df.shape


# In[9]:


# Now remove some of the outliers
# Plot the histogram of passenger_count
#plt.hist(df['passenger_count'], bins=np.arange(0, 10))


# In[10]:


#Passengers count should range from 1 to 7, rest all are outliers let's remove them
df = df[(df.passenger_count > 0) & (df.passenger_count < 8)]


# In[11]:


df.shape


# In[12]:


# Lets consider the histogram of fare_amount
df[df.fare_amount<500].fare_amount.hist(bins=100, figsize=(14,3))


# In[13]:


df.shape


# In[14]:


# As we can see we have outliers for fare_amount > 60. Remove them from data set
df = df[(df.fare_amount > 0) & (df.fare_amount <= 60)]


# In[15]:


df.shape


# In[16]:


# Now we now that new york city had latitude around 40.71 and longitude around -74.00 (reference : http://latitudelongitude.org/us/new-york-city/).
# So taking a rough estimate remove outliers out of range
df = df[(df.pickup_longitude > -75) & (df.pickup_longitude < -70)]
df = df[(df.dropoff_longitude > -75) & (df.dropoff_longitude < -70)]
df = df[(df.pickup_latitude > 38) & (df.pickup_latitude < 43)]
df = df[(df.dropoff_latitude > 38) & (df.dropoff_latitude < 43)]


# In[17]:


df.shape


# In[18]:


#Plotting scatter plot of longitude and latitude
df.plot(kind='scatter',x='pickup_longitude',y='pickup_latitude',c='blue',s=0.2,alpha=.6)


# In[19]:


# Remove more outliers of latitude and longitude
df = df[(df.pickup_longitude > -74.30) & (df.pickup_longitude < -73.00)]
df = df[(df.dropoff_longitude > -74.30) & (df.dropoff_longitude < -73.00)]
df = df[(df.pickup_latitude > 40.40) & (df.pickup_latitude < 41.71)]
df = df[(df.dropoff_latitude > 40.40) & (df.dropoff_latitude < 41.71)]


# In[20]:


df.shape


# In[21]:


df.plot(kind='scatter',x='pickup_longitude',y='pickup_latitude',c='blue',s=0.2,alpha=.6)


# In[22]:


# Remove more outliers of latitude and longitude
df = df[(df.pickup_longitude > -74.05) & (df.pickup_longitude < -73.75)]
df = df[(df.dropoff_longitude > -74.05) & (df.dropoff_longitude < -73.75)]
df = df[(df.pickup_latitude > 40.55) & (df.pickup_latitude < 40.85)]
df = df[(df.dropoff_latitude > 40.55) & (df.dropoff_latitude < 40.85)]


# In[23]:


df.plot(kind='scatter',x='pickup_longitude',y='pickup_latitude',c='blue',s=0.2,alpha=.99)


# In[24]:


#Lets calculate distance travelled in each trip using HaverSine distance
#Reference : https://community.esri.com/groups/coordinate-reference-systems/blog/2017/10/05/haversine-formula
def cal_haversinedistance(pickup_lat : float, pickup_long : float, dropoff_lat : float, dropoff_long : float):
    R = 6373.0
    long_diff = dropoff_long - pickup_long
    lat_diff = dropoff_lat - pickup_lat
    
    a = sin(lat_diff / 2)**2 + cos(pickup_lat) * cos(dropoff_lat) * sin(long_diff / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance


# In[25]:


def dist_calc(df):
    R = 6373.0
    
    for i,row in df.iterrows():

        pickup_lat = radians(row['pickup_latitude'])
        pickup_long = radians(row['pickup_longitude'])
        dropoff_lat = radians(row['dropoff_latitude'])
        dropoff_long = radians(row['dropoff_longitude'])

        distance = cal_haversinedistance(pickup_lat, pickup_long, dropoff_lat, dropoff_long)
        df.at[i,'distance'] = distance


# In[26]:


dist_calc(df)


# In[27]:


df.head()


# In[28]:


df.describe()


# In[29]:


df = df[~((df.fare_amount > 40) & (df.distance < 5))]


# In[30]:


# Feature Engineering
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
df['hour'] = df['pickup_datetime'].dt.hour
df['day'] = df['pickup_datetime'].dt.day
df['week'] = df['pickup_datetime'].dt.week
df['month'] = df['pickup_datetime'].dt.month
df['day_of_year'] = df['pickup_datetime'].dt.dayofyear
df['week_of_year'] = df['pickup_datetime'].dt.weekofyear
df['year'] = df['pickup_datetime'].dt.year


# In[31]:


# There is no outlier in hour, day, week, month, day_of_year, week_of_year as evident from their min and max values
df.describe()


# In[32]:


# Let's compute correlation coefficient 
df.corr()


# In[33]:


# Let's check some relationships:
# (1) fare_amount and distance
df.plot(kind='scatter',x='distance',y='fare_amount',c='blue',s=0.2,alpha=.6)


# In[34]:


df['fare_amount'].corr(df['distance'])
# Linear Plot and correlation factor of 0.88 signifies linear relationship between fare_amount and distance travelled in a trip. The fare gets increased with the distance travelled.


# In[35]:


# (2) fare_amount and day of time
df[['fare_amount', 'hour']].boxplot(by = 'hour', showfliers = False)


# In[36]:


df['fare_amount'].corr(df['hour'])
# Findings : 1) Correlation factor = -0.01607561723255814 which does not implies any relation betweem fare_amount and hour.
# max fare_amount is in the morning at 5 am which keeps on decreasing till 7 am and it remains almost constant theroughout the day. Morning high fares 
#may be due to the fact of people going to/from airport.


# In[37]:


# (3) distance and day of time
df[['distance', 'hour']].boxplot(by = 'hour', showfliers = False)


# In[38]:


df['distance'].corr(df['hour'])
# Findings : 1) Correlation factor = -0.029914726054103426 which does not implies any relation betweem distance and hour.
# max_distance travelled is in the morning at 5 am may be due to the same fact of taxi going to/from airport. So there is no significant relation betweem distance and day of time. 


# In[39]:


df[['fare_amount', 'month']].boxplot(by = 'month', showfliers = False)
#It shows the maximum fare received in the months of may and June due to summer vacations. Also the maximum fare is received in the months of October
#to december may be to the new year.


# In[40]:


plt.scatter(x=df['day'], y=df['fare_amount'], s=1.5)
#No correlation between date and fare_amount


# In[41]:


def populateAirportDataSet(data):
    jfk_coord = (40.639722, -73.778889)
    ewr_coord = (40.6925, -74.168611)
    lga_coord = (40.77725, -73.872611)
    mylist = []
    for i,row in data.iterrows():
        pickup_lat = radians(row['pickup_latitude'])
        dropoff_lat = radians(row['dropoff_latitude'])
        pickup_lon = radians(row['pickup_longitude'])
        dropoff_lon = radians(row['dropoff_longitude'])
        #print('vivek')
        pickup_jfk = cal_haversinedistance(pickup_lat, pickup_lon, jfk_coord[0], jfk_coord[1]) 
        dropoff_jfk = cal_haversinedistance(jfk_coord[0], jfk_coord[1], dropoff_lat, dropoff_lon) 
        pickup_ewr = cal_haversinedistance(pickup_lat, pickup_lon, ewr_coord[0], ewr_coord[1])
        dropoff_ewr = cal_haversinedistance(ewr_coord[0], ewr_coord[1], dropoff_lat, dropoff_lon) 
        pickup_lga = cal_haversinedistance(pickup_lat, pickup_lon, lga_coord[0], lga_coord[1]) 
        dropoff_lga = cal_haversinedistance(lga_coord[0], lga_coord[1], dropoff_lat, dropoff_lon) 
        if(pickup_jfk < 2 or dropoff_jfk < 2 or pickup_ewr < 2 or dropoff_ewr < 2 or pickup_lga < 2 or dropoff_lga < 2):
            mylist.append(1)
        else:
            mylist.append(0)
    return mylist


# In[42]:


df['isAirportRide'] = populateAirportDataSet(df)


# In[43]:


df.describe()


# In[44]:


df.plot(kind='scatter',x='hour',y='fare_amount',c='blue',s=0.2,alpha=.99)


# In[45]:


test = pd.read_csv('test.csv')


# In[46]:


USFederal_Holidays = USFederalHolidayCalendar().holidays(start='2005-01-01', end='2017-12-31').to_pydatetime()

USFederal_Holidays = USFederal_Holidays.tolist()

def IsUSFederalHoliday(data):
    date = data['pickup_datetime']
    if date in USFederal_Holidays:
        return 1
    else:
        
        return 0


# In[47]:


df['is_holiday'] = df.apply(IsUSFederalHoliday,axis = 1)


# In[48]:


# Modelling of data


# In[49]:


feature_names = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count', 'distance', 'hour', 'year', 'isAirportRide', 'is_holiday']
label_name = 'fare_amount'
x_train = df[feature_names]
y_train = df[label_name]
test = pd.read_csv('test.csv')
dist_calc(test)
test['hour'] = pd.to_datetime(test['pickup_datetime']).dt.hour
test['year'] = pd.to_datetime(test['pickup_datetime']).dt.year
test['isAirportRide'] = populateAirportDataSet(test)
test['is_holiday'] = test.apply(IsUSFederalHoliday,axis = 1)


# In[50]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression


# In[51]:


#Random Forest Model
x_train1, x_test1, y_train1, y_test1 = train_test_split(x_train,y_train, test_size=0.2)

rfr = RandomForestRegressor()
rfr.fit(x_train1, y_train1)
rfr_prediction = rfr.predict(x_test1)

#rfr_submission = pd.DataFrame({"key": test['key'],"fare_amount": rfr_prediction},columns = ['key','fare_amount'])
rmse = np.sqrt(mean_squared_error(rfr_prediction,y_test1))
print ("root mean Squared Error : {}".format(rmse))


# In[52]:


xtest = test[feature_names]
rfr.fit(x_train, y_train)
rfr_prediction = rfr.predict(xtest)


# In[53]:


submission = pd.read_csv('sample_submission.csv')
submission['fare_amount'] = rfr_prediction
submission.to_csv('vivek_submission.csv', index=False)


# In[54]:


# Linear Regression
x_train1, x_test1, y_train1, y_test1 = train_test_split(x_train,y_train, test_size=0.2)
linear_reg = LinearRegression()
linear_reg.fit(x_train1, y_train1)


# In[55]:


print(linear_reg.coef_)


# In[56]:


y_pred = linear_reg.predict(x_test1)


# In[57]:


rmse = np.sqrt(mean_squared_error(y_pred, y_test1))
print ("root mean Squared Error : {}".format(rmse))

