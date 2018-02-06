
# coding: utf-8

# #### Chicago Cab Analysis
# 
# I had to come up with some real world data to analyze for my final Data Mining project. I had read about Uber & Lyft driving the regular cab drivers out of business & this intrigued me. Chicago has the second largest cab fleets & according to "rumors" they were facing extinction. According to reports, approximately 40% of the 7000 strong cab fleet had not had a fare for the month of March 2017 & I wanted to see this for myself.
# 
# I came across the Chicago cab data through the Chicago Data Portal. Their data was from years 2013-2016 & was of size 40 GB & had around 127 million records. I extracted the data using an API into my Python notebook. Unfortunately, due to system restrictions, I was able to work on only a small sample set of this data (~1%). I analyzed this sample data & my findings resonated with what some articles mentioned about how cab drivers are facing unemployment.
# 

# In[1]:

#!pip install sodapy
#!pip install geopy

import pandas as pd
import numpy as np
import matplotlib as matpp
import matplotlib.pyplot as pp
import matplotlib.colors as clr
from sodapy import Socrata
import time
import datetime
import folium
import calendar

client = Socrata("data.cityofchicago.org", None)


# The complete dataset is available at the [Chicago Data Portal](https://data.cityofchicago.org/Transportation/Taxi-Trips/wrvz-psew/data) has more data than what my system could handle. It didn't make sense to download a 40 GB csv file, read it & then analyze it. Hence I accessed the data through the Socrata Open Data API (SODA). System restrictions meant that I could not access the entire dataset. Hence I decided to subset the data. I took a sample of each of the 3 years for sizes 100k, 250k & 500k, structured & aggregated the data & plotted them.

# In[2]:

sample_100k_2016 = client.get("wrvz-psew",limit=100000,
                              where="trip_start_timestamp > '2016-01-01T00:00:00' \
                              AND trip_start_timestamp <= '2016-12-31T00:00:00' AND \
                              pickup_census_tract IS NOT NULL AND \
                              dropoff_census_tract IS NOT NULL AND \
                              company IS NOT NULL")


# In[3]:

sample_250k_2016 = client.get("wrvz-psew",limit=250000,
                              where="trip_start_timestamp > '2016-01-01T00:00:00' \
                              AND trip_start_timestamp <= '2016-12-31T00:00:00' AND \
                              pickup_census_tract IS NOT NULL AND \
                              dropoff_census_tract IS NOT NULL AND \
                              company IS NOT NULL")


# In[4]:

sample_500k_2016 = client.get("wrvz-psew",limit=500000,
                              where="trip_start_timestamp > '2016-01-01T00:00:00' \
                              AND trip_start_timestamp <= '2016-12-31T00:00:00' AND \
                              pickup_census_tract IS NOT NULL AND \
                              dropoff_census_tract IS NOT NULL AND \
                              company IS NOT NULL")


# In[5]:

# converting to a dataframe

chicago_cab_data_2016_sample_100k = pd.DataFrame.from_records(sample_100k_2016).iloc[:,[14,21]]
chicago_cab_data_2016_sample_250k = pd.DataFrame.from_records(sample_250k_2016).iloc[:,[14,21]]
chicago_cab_data_2016_sample_500k = pd.DataFrame.from_records(sample_500k_2016).iloc[:,[14,21]]

# tagging, appending & restructuring the data

chicago_cab_data_2016_sample_100k['sample']='sample_100k'
chicago_cab_data_2016_sample_250k['sample']='sample_250k'
chicago_cab_data_2016_sample_500k['sample']='sample_500k'

chicago_appended_sample_data=chicago_cab_data_2016_sample_100k.append(chicago_cab_data_2016_sample_250k,ignore_index=True).append(chicago_cab_data_2016_sample_500k,ignore_index=True)

chicago_appended_sample_data["trip_date"]=chicago_appended_sample_data["trip_start_timestamp"].astype(str)
chicago_appended_sample_data["trip_start_time"]=chicago_appended_sample_data["trip_date"].str[-8:]
chicago_appended_sample_data["trip_date"]=chicago_appended_sample_data["trip_date"].str[:10]
chicago_appended_sample_data["trip_year"]=chicago_appended_sample_data["trip_date"].str[:4]
chicago_appended_sample_data["trip_month"]=chicago_appended_sample_data["trip_date"]
chicago_appended_sample_data["trip_month"]=pd.to_datetime(chicago_appended_sample_data["trip_month"])
chicago_appended_sample_data["trip_month"]=chicago_appended_sample_data["trip_month"].apply(lambda x: datetime.datetime.strftime(x,'%b'))


# In[6]:

# comparing samples of different sizes for 2016

sample_data_grouped=pd.pivot_table(chicago_appended_sample_data,values='taxi_id',index=('trip_month'),columns=('sample'),fill_value="",margins=2,margins_name="total",aggfunc=[len])
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
sample_data_grouped=sample_data_grouped.reindex(months)
sample_data_grouped = sample_data_grouped.xs('len', axis=1, drop_level=True)
sample_data_grouped


# In[7]:

# plotting samples of different sizes for 2016

year_on_year_sample_data_grouped = pd.DataFrame(sample_data_grouped.iloc[0:12,0:3])

color=["TEAL", "CRIMSON", "NAVY"]
year_on_year_sample_data_grouped.plot(figsize=(10,5),color=color,linewidth=2)
pp.xlabel('rides trend with different sample sizes',color='black')
pp.xticks(color = 'black')
pp.yticks(color = 'black')

pp.ylabel('total rides')
pp.title('ride trend across months (2016)',color='black')
pp.xticks(rotation='horizontal')
pp.show()


# The above graph consists of 3 sample sets of 100k, 250k & 500k for the year 2016. Looking at the samples, it seems rather obvious that the samples follow a certain trend & that they can be considered to be an approximate but similar representation of the population.
# I opted for the middle value of 250k & proceeded with my analysis for the years 2014-16.

# In[8]:

chicago_cab_data_raw_2016 = client.get("wrvz-psew",limit=250000,where="trip_start_timestamp > '2016-01-01T00:00:00' AND trip_start_timestamp <= '2016-12-31T00:00:00' AND dropoff_centroid_longitude IS NOT NULL AND dropoff_centroid_latitude IS NOT NULL AND pickup_centroid_longitude IS NOT NULL AND pickup_centroid_latitude IS NOT NULL AND company IS NOT NULL")
chicago_cab_data_raw_2015 = client.get("wrvz-psew",limit=250000,where="trip_start_timestamp > '2015-01-01T00:00:00' AND trip_start_timestamp <= '2015-12-31T00:00:00' AND dropoff_centroid_longitude IS NOT NULL AND dropoff_centroid_latitude IS NOT NULL AND pickup_centroid_longitude IS NOT NULL AND pickup_centroid_latitude IS NOT NULL AND company IS NOT NULL")
chicago_cab_data_raw_2014 = client.get("wrvz-psew",limit=250000,where="trip_start_timestamp > '2014-01-01T00:00:00' AND trip_start_timestamp <= '2014-12-31T00:00:00' AND dropoff_centroid_longitude IS NOT NULL AND dropoff_centroid_latitude IS NOT NULL AND pickup_centroid_longitude IS NOT NULL AND pickup_centroid_latitude IS NOT NULL AND company IS NOT NULL")

chicago_cab_data_2016 = pd.DataFrame.from_records(chicago_cab_data_raw_2016)
chicago_cab_data_2015 = pd.DataFrame.from_records(chicago_cab_data_raw_2015)
chicago_cab_data_2014 = pd.DataFrame.from_records(chicago_cab_data_raw_2014)


# In[9]:

chicago_3_year_raw_data = pd.concat([chicago_cab_data_2014,chicago_cab_data_2015,chicago_cab_data_2016])
chicago_3_year_raw_data.head(2)


# In[10]:

chicago_3_year_raw_data=chicago_3_year_raw_data.iloc[:,[14,18,0,10,12,11,13,2,4,3,5,21,17,19,20,7,6,15,16,22,8]]
chicago_3_year_raw_data["trip_date"]=pd.to_datetime(chicago_3_year_raw_data["trip_start_timestamp"])
chicago_3_year_raw_data.head(2)


# In[11]:

# manipulating trip_date column to extract day & time 

chicago_3_year_raw_data["trip_date"]=chicago_3_year_raw_data["trip_date"].astype(str)
chicago_3_year_raw_data["trip_start_time"]=chicago_3_year_raw_data["trip_date"].str[-8:]
chicago_3_year_raw_data["trip_date"]=chicago_3_year_raw_data["trip_date"].str[:10]
chicago_3_year_raw_data["trip_year"]=chicago_3_year_raw_data["trip_date"].str[:4]
chicago_3_year_raw_data["trip_month"]=chicago_3_year_raw_data["trip_date"]

chicago_3_year_raw_data.head(2)


# In[12]:

# getting month name from month number

chicago_3_year_raw_data["trip_month"]=pd.to_datetime(chicago_3_year_raw_data["trip_month"])
chicago_3_year_raw_data["trip_month"]=chicago_3_year_raw_data["trip_month"].apply(lambda x: datetime.datetime.strftime(x,'%b'))
chicago_3_year_raw_data.tail(2)


# In[13]:

# month on month taxi ride counts across years

chicago_3_year_structured=pd.pivot_table(chicago_3_year_raw_data,values='taxi_id',index=('trip_month'),columns=('trip_year'),fill_value="",margins=2,margins_name="total",aggfunc=[len])
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
chicago_3_year_structured=chicago_3_year_structured.reindex(months)
chicago_3_year_structured


# In[14]:

# dropping level for ease

chicago_3_year_structured = chicago_3_year_structured.xs('len', axis=1, drop_level=True)
chicago_3_year_structured


# In[15]:

chicago_3_year_structured = pd.DataFrame(chicago_3_year_structured.iloc[0:12,0:3])
chicago_3_year_structured


# In[16]:

# plotting month on month total ride count across years

color=["MEDIUMBLUE", "CRIMSON", "DARKGREEN"]

chicago_3_year_structured.plot(figsize=(12,5),color=color,linewidth=2)

#ax.set_xticks(chicago_3_year_structured['trip_month'],minor=True)
pp.xlabel('month-on-month ride trend',color='black')

pp.xticks(color = 'black')
pp.yticks(color = 'black')
pp.ylabel('total rides',color='black')
pp.title('ride trend (2014-2016)',color='black')
pp.xticks(rotation='horizontal')
pp.show()


# As we can see, the number of rides go up in each of the 3 years as each year begins. However, as the second half of the year begins, the ride numbers go down drastically. 
# 
# Are cab drivers really losing business due to Uber & lyft?

# In[17]:

taxis_in_business=chicago_3_year_raw_data.groupby(["trip_year","trip_month"]).agg({"taxi_id": pd.Series.nunique})
taxis_in_business=taxis_in_business.reset_index()

taxis_in_business['year_month'] = taxis_in_business["trip_year"] +" - "+taxis_in_business["trip_month"]
taxis_in_business=taxis_in_business.iloc[:,2:]

year_month=['2014 - Jan','2014 - Feb','2014 - Mar','2014 - Apr','2014 - May','2014 - Jun','2014 - Jul','2014 - Aug','2014 - Sep','2014 - Oct','2014 - Nov','2014 - Dec','2015 - Jan','2015 - Feb','2015 - Mar','2015 - Apr','2015 - May','2015 - Jun','2015 - Jul','2015 - Aug','2015 - Sep','2015 - Oct','2015 - Nov','2015 - Dec','2016 - Jan','2016 - Feb','2016 - Mar','2016 - Apr','2016 - May','2016 - Jun','2016 - Jul','2016 - Aug','2016 - Sep','2016 - Oct','2016 - Nov','2016 - Dec']

taxis_in_business=taxis_in_business.set_index('year_month')
taxis_in_business=taxis_in_business.reindex(year_month)

taxis_in_business = taxis_in_business.rename(columns={'taxi_id':'active taxis'})

taxis_in_business.plot(figsize=(12,5),color = 'purple',linewidth = 3,kind="line")

pp.xlabel('year - months (2014-2016)',color='black')
pp.xticks(color = 'black')
pp.yticks(color = 'black')
pp.ylabel('total active taxis',color='black')
pp.title('taxis in business',color='black')
pp.xticks(rotation='vertical')
pp.legend(loc='center left', bbox_to_anchor=(1,.96))
print(taxis_in_business)
pp.show()


# As we can see, the unique count of taxi id has gone down from ~2500 to ~1800 over a period of 3 years. Which is almost a 70% decrease - which resonates with the below mentioned article. [Number of Chicago taxi drivers hits 10-year low as ride-share companies take off - Chicago Tribune (Dec 16, 2017)](http://www.chicagotribune.com/news/ct-chicago-taxi-driver-decline-met-20161214-story.html)
# 
# I then wanted to look at the daily ride trend across the entire timeline hoping to find some interesting trends.

# In[18]:

# day on day total rides across 2014-16

day_on_day_trend_2016 = chicago_3_year_raw_data[(chicago_3_year_raw_data["trip_date"] >= '2014-01-01') & (chicago_3_year_raw_data["trip_date"] <= '2016-12-31')]
day_on_day_trend_2016 = day_on_day_trend_2016.rename(columns={'trip_id':'count of trips'})
ax = pd.pivot_table(day_on_day_trend_2016, values='count of trips', columns=['trip_date'], fill_value="",aggfunc=[len]).T
ax = ax.xs('len', axis=0, drop_level=True)


# In[19]:

ax.plot(figsize=(20,6),color='red')
pp.xlabel('daily ride trend',color='black')
pp.xticks(color = 'black')
pp.yticks(color = 'black')
pp.ylabel('total rides',color='black')
pp.title('ride trend across the year',color='black')
pp.xticks(rotation='vertical')
pp.legend(loc='center left', bbox_to_anchor=(1,.97))
pp.show()


# As we can see in the above image, there are distinct peaks & troughs across the entire timeline. However, for each year, there is a distinct peak at around March. To get a better idea of the data, I aggregated the data on "total rides on a per day basis", sorted the data & looked at the top 10 days with most number of cab rides. Also, using the date, I extracted the days corresponding to these dates.

# In[20]:

# top 10 days with most cab rides (2014-16)

chicago_3_year_raw_data['count']=1
reqd_cols=chicago_3_year_raw_data.pivot_table(columns="trip_date",aggfunc=sum).T.reset_index().sort_values(by='count',ascending=False)
reqd_cols['trip_date']=reqd_cols['trip_date'].astype('datetime64[ns]')
reqd_cols['day']=reqd_cols['trip_date'].dt.strftime('%A')
reqd_cols['year']=reqd_cols['trip_date'].dt.strftime('%Y')
reqd_cols.head(10)


# One interesting observation was that in the top ten days with most cabs hired, each and every day was either on a Friday or a Saturday. So one hypothesis that I could come up with is that the weekend, the Chicagoans prefer to take cabs over the weekend more than the weekday. However, this would need more clarity. So, I plotted the cab hire counts across the entire week for each of the 3 years. 

# In[21]:

# day wise cabs hires

day_wise_cab_rides=reqd_cols.pivot_table(index="day",columns="year",values="count",aggfunc=np.mean)
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday','Sunday']
day_wise_cab_rides=day_wise_cab_rides.reindex(days)
day_wise_cab_rides

color = ["coral","red","darkred"]
day_wise_cab_rides.plot(kind="bar",figsize=(12,8),legend=False,color=color)
pp.xlabel('average cab hires across days (2014-16)',color='black')
pp.xticks(color = 'black')
pp.yticks(color = 'black')
pp.ylabel('average number of rides daywise',color='black')
pp.title('ride trend across the week',color='black')
pp.xticks(rotation='vertical')
pp.legend(loc='center left', bbox_to_anchor=(1,.945))
pp.show()


# As evident from the graph above, my hypothesis held true. Weekday rides are low as compared to the weekend & as the weekend approaches, the ride count keeps increasing.
# 
# I then wanted to see if the there has been a significant change in average ride costs or average ride times.

# In[22]:

# average month on month ride cost (2014-16) 

chicago_3_year_raw_data1=chicago_3_year_raw_data.dropna()
chicago_3_year_raw_data1[["fare","tips","trip_total","extras","tolls","trip_miles","trip_seconds"]]=chicago_3_year_raw_data1[["fare","tips","trip_total","extras","tolls","trip_miles","trip_seconds"]].apply(pd.to_numeric)

chicago_3_year_structured_money=pd.pivot_table(chicago_3_year_raw_data1,values=('trip_total'),index=('trip_month'),columns=('trip_year'),margins=2,margins_name="total",aggfunc=[np.average])
chicago_3_year_structured_money=chicago_3_year_structured_money.round(2)
chicago_3_year_structured_money=chicago_3_year_structured_money.iloc[:12,:3]
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
chicago_3_year_structured_money=chicago_3_year_structured_money.reindex(months)
chicago_3_year_structured_money.columns = chicago_3_year_structured_money.columns.set_levels(['avg ride cost (in USD)'], level=0)
chicago_3_year_structured_money


# In[23]:

chicago_3_year_structured_money_1 = chicago_3_year_structured_money.xs('avg ride cost (in USD)', axis=1, drop_level=True)
chicago_3_year_structured_money_1 = chicago_3_year_structured_money_1.unstack()


# In[24]:

chicago_3_year_structured_money_1.plot(figsize=(15,5),color='red')

pp.xlabel('average cost of cab rides (2014-16)',color='black')
pp.xticks(color = 'black')
pp.yticks(color = 'black')
pp.ylabel('average cost of rides (in USD)',color='black')
pp.title('average trip cost across the years',color='black')
pp.xticks(rotation='vertical')

pp.show()


# As we can see, the average ride cost at the start of the year is slightly less than $ 12. However, just before 2016 ended, the average ride cost was around $ 16 before reducogm to around $ 12. 

# In[25]:

# average month on month trip duration in minutes (2014-16)

chicago_3_year_structured_duration=pd.pivot_table(chicago_3_year_raw_data1,values=('trip_seconds'),index=('trip_month'),columns=('trip_year'),margins=2,margins_name="total",aggfunc=[np.average])
chicago_3_year_structured_duration=(chicago_3_year_structured_duration/60).round(2)
chicago_3_year_structured_duration=chicago_3_year_structured_duration.iloc[:12,:3]
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
chicago_3_year_structured_duration=chicago_3_year_structured_duration.reindex(months)
chicago_3_year_structured_duration.columns = chicago_3_year_structured_duration.columns.set_levels(['avg trip time (in minutes)'], level=0)
chicago_3_year_structured_duration


# In[26]:

chicago_3_year_structured_duration_1 = chicago_3_year_structured_duration.xs('avg trip time (in minutes)', axis=1,drop_level=True)
chicago_3_year_structured_duration_1 =chicago_3_year_structured_duration_1.unstack()


# In[27]:

chicago_3_year_structured_duration_1.plot(figsize=(15,5),color='green')

pp.xlabel('average duration of cab rides (2014-16)',color='black')
pp.xticks(color = 'black')
pp.yticks(color = 'black')
pp.ylabel('average duration of rides (in minutes)',color='black')
pp.title('average trip duration across the years',color='black')
pp.xticks(rotation='vertical')

pp.show()


# As we can see, the average trip duration at the start of the year is around 11.5 minutes. However, just before 2016 ended, the average ride cost was around 13 minutes.
# 
# The above 2 data points do not make much sense when looked at individually. So, I merged these 2 datasets & plotted the data overlapped data points.

# In[28]:

# average month on month across years ride cost vs trip duration

compare_trip_cost_and_duration = [chicago_3_year_structured_money, chicago_3_year_structured_duration]
compare_trip_cost_and_duration_1 = pd.concat(compare_trip_cost_and_duration, axis=1)
compare_trip_cost_and_duration_1


# In[29]:

# restructuring data for plotting purposes

compare_trip_cost_and_duration_1=compare_trip_cost_and_duration_1.stack()
compare_trip_cost_and_duration_1=compare_trip_cost_and_duration_1.reset_index()

compare_trip_cost_and_duration_1['year_month'] = compare_trip_cost_and_duration_1["trip_year"] +" - "+compare_trip_cost_and_duration_1["trip_month"]
compare_trip_cost_and_duration_1=compare_trip_cost_and_duration_1.iloc[:,2:]

compare_trip_cost_and_duration_1=compare_trip_cost_and_duration_1.set_index('year_month')
compare_trip_cost_and_duration_1.head(5)


# In[30]:

year_month=['2014 - Jan','2014 - Feb','2014 - Mar','2014 - Apr','2014 - May','2014 - Jun','2014 - Jul','2014 - Aug','2014 - Sep','2014 - Oct','2014 - Nov','2014 - Dec','2015 - Jan','2015 - Feb','2015 - Mar','2015 - Apr','2015 - May','2015 - Jun','2015 - Jul','2015 - Aug','2015 - Sep','2015 - Oct','2015 - Nov','2015 - Dec','2016 - Jan','2016 - Feb','2016 - Mar','2016 - Apr','2016 - May','2016 - Jun','2016 - Jul','2016 - Aug','2016 - Sep','2016 - Oct','2016 - Nov','2016 - Dec']
compare_trip_cost_and_duration_1=compare_trip_cost_and_duration_1.reindex(year_month)
compare_trip_cost_and_duration_1.head(5)


# In[31]:

compare_trip_cost_and_duration_1.plot(figsize=(15,5),color=['RED','GREEN'])

pp.xlabel('month on month - (2014-16)',color='black')
pp.xticks(color = 'black')
pp.yticks(color = 'black')
pp.title('avg cost of cab rides v/s avg trip duration ',color='black')
pp.xticks(rotation='vertical')

pp.show()


# As we can see initially, the 2 variables, average cab ride times & average cost of taxi hires go hand in hand. However, as the year 2016 begins, even though average ride time has remained a more or less a consistent value, the average cab ride cost has increased tremendously!  
# 
# Then to check if there was any particluar cab ride fare method that was more common than others, I analysed the payment type field of this dataset.

# In[32]:

# credit card payments vs cash payments

chicago_3_year_raw_data
type_of_payment = chicago_3_year_raw_data[["trip_id","payment_type","trip_year","trip_month"]]
type_of_payment_grouped = type_of_payment.pivot_table(index="trip_month",columns=["trip_year","payment_type"],values="trip_id",aggfunc=np.count_nonzero,margins=0) 

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
type_of_payment_grouped=type_of_payment_grouped.reindex(months)
type_of_payment_grouped=type_of_payment_grouped.drop(['Dispute','No Charge','Unknown'],axis=1,level=1)


# In[33]:

type_of_payment_grouped


# In[34]:

type_of_payment_percent = round(100*type_of_payment_grouped.div(type_of_payment_grouped.sum(axis=1, level=0), level=0),1)
type_of_payment_percent_1 = type_of_payment_percent.astype(str)+"%"
type_of_payment_percent_1


# In[35]:

type_of_payment_grouped_yoy = type_of_payment.pivot_table(index=["trip_year","trip_month"],columns="payment_type",values="trip_id",aggfunc=np.count_nonzero,margins=0) 
type_of_payment_grouped_yoy=type_of_payment_grouped_yoy.iloc[:,:2]
type_of_payment_grouped_yoy=type_of_payment_grouped_yoy.reset_index()
type_of_payment_grouped_yoy['year_month'] = type_of_payment_grouped_yoy["trip_year"] +" - "+type_of_payment_grouped_yoy["trip_month"]
type_of_payment_grouped_yoy=type_of_payment_grouped_yoy.iloc[:,2:]
type_of_payment_grouped_yoy=type_of_payment_grouped_yoy.set_index('year_month')
year_month=['2014 - Jan','2014 - Feb','2014 - Mar','2014 - Apr','2014 - May','2014 - Jun','2014 - Jul','2014 - Aug','2014 - Sep','2014 - Oct','2014 - Nov','2014 - Dec','2015 - Jan','2015 - Feb','2015 - Mar','2015 - Apr','2015 - May','2015 - Jun','2015 - Jul','2015 - Aug','2015 - Sep','2015 - Oct','2015 - Nov','2015 - Dec','2016 - Jan','2016 - Feb','2016 - Mar','2016 - Apr','2016 - May','2016 - Jun','2016 - Jul','2016 - Aug','2016 - Sep','2016 - Oct','2016 - Nov','2016 - Dec']
type_of_payment_grouped_yoy=type_of_payment_grouped_yoy.reindex(year_month)

type_of_payment_grouped_yoy.plot(figsize=(15,5),color=['navy','orange'],linewidth=2)

pp.xlabel('month on month - (2014-16)',color='black')
pp.ylabel('payments types',color='black')

pp.xticks(color = 'black')
pp.yticks(color = 'black')
pp.xticks(rotation='vertical')

pp.title("Cash v/s Credit Card Payments (2014 - 2016)",color ='black')

pp.show()


# At the start of 2014, Cash payments contributed to approximately 73%. However, over time of 3 years, we can see that payment through cash reduced to approximately 56% & payment through Credit card increased in the same time period.
# 
# I wanted to have a look at the most common pickup/drop-off spots for cab drivers. In order to do that I had to manipulate the latitudes and longitudes into formats that would be acceptable by the package geopy. I passed the latitude, longitude values through the geopy function & reverse geolocated the addresses of the pickup/ drop-offs on a map using the folium package. Basically, I created a function that enables the user to look up any number of top pickup/drop off places in Chicago. The size of the bubble indicates the count of the number of pickup/ drop offs.

# In[36]:

# manipulating data for plotting pick up & drop offs on a map

chicago_3_year_raw_data['pick_up_lat_lon']=chicago_3_year_raw_data[['pickup_centroid_latitude','pickup_centroid_longitude']].apply(tuple,axis=1)
chicago_3_year_raw_data['drop_off_lat_lon']=chicago_3_year_raw_data[['dropoff_centroid_latitude','dropoff_centroid_longitude']].apply(tuple,axis=1)
chicago_3_year_raw_data.dtypes


# In[37]:

top_drop_off_spots=pd.crosstab(index=(chicago_3_year_raw_data['drop_off_lat_lon']),columns='count')
top_drop_off_spots=top_drop_off_spots.sort_values(ascending=False,by='count').head(10)
top_drop_off_spots=top_drop_off_spots.reset_index()
top_drop_off_spots[['lat','lon']]=top_drop_off_spots['drop_off_lat_lon'].apply(pd.Series)

# formatting lat & lon for drop offs

top_pick_up_spots=pd.crosstab(index=(chicago_3_year_raw_data['pick_up_lat_lon']),columns='count')
top_pick_up_spots=top_pick_up_spots.sort_values(ascending=False,by='count').head(10)
top_pick_up_spots=top_pick_up_spots.reset_index()
top_pick_up_spots[['lat','lon']]=top_pick_up_spots['pick_up_lat_lon'].apply(pd.Series)


# In[38]:

from geopy.geocoders import Nominatim
geolocator = Nominatim()


# In[39]:

# Top areas where passengers are picked up  

def top_pickup(trip_year,counter):
    
    top_pick_up_spots=pd.pivot_table(data=chicago_3_year_raw_data,values='taxi_id',index=[('pick_up_lat_lon')],columns=['trip_year'],aggfunc='count',fill_value=0,margins=2,margins_name="total")
    top_pick_up_spots=top_pick_up_spots.loc[:,trip_year]
    top_pick_up_spots=pd.DataFrame(top_pick_up_spots)
    top_pick_up_spots=top_pick_up_spots.drop('total').sort_values(by=trip_year,ascending=False).head(counter)
    top_pick_up_spots=top_pick_up_spots.reset_index()
    top_pick_up_spots[['lat','lon']]=top_pick_up_spots['pick_up_lat_lon'].apply(pd.Series)
    top_pick_up_spots_1=pd.DataFrame(top_pick_up_spots.drop('pick_up_lat_lon',axis=1))
    top_pick_up_spots_1=top_pick_up_spots.iloc[:,1:]
    
    top_pick_up_spots_1['address'] = top_pick_up_spots_1.apply(lambda row: geolocator.reverse((row['lat'], row['lon'])), axis=1)
    
    top_pick_up_spots_1['lat']=pd.DataFrame.convert_objects(top_pick_up_spots_1['lat'],convert_numeric=True)
    top_pick_up_spots_1['lon']=pd.DataFrame.convert_objects(top_pick_up_spots_1['lon'],convert_numeric=True)
    top_pick_up_spots_1['address']=top_pick_up_spots_1['address'].astype(str)
    
    lat=top_pick_up_spots_1['lat']
    lon=top_pick_up_spots_1['lon']
    
    lat_mean = top_pick_up_spots_1['lat'].mean()
    lon_mean = top_pick_up_spots_1['lon'].mean()
    
    locationlist = top_pick_up_spots_1[["lat","lon"]].values.tolist()
    labels = top_pick_up_spots_1["address"].values.tolist()

    chimap = folium.Map([lat_mean,lon_mean], zoom_start=12)
    folium.TileLayer('cartodbpositron').add_to(chimap)

    pick_up_size=top_pick_up_spots_1.iloc[:,0]    
        
    for i in range(len(locationlist)):
        popup = folium.Popup(str(labels[i]) +" -> (total pick ups  = "+ str(int(pick_up_size[i])) + ")", parse_html=True)
        icon=folium.Icon(color='red')
        folium.CircleMarker(locationlist[i], popup=popup, icon = icon,color='green',fill_color='green',radius = pick_up_size[i]/1000,fill=True).add_to(chimap)
    return(chimap)
    
top_pickup('2015',10)


# In[40]:

# Top areas to where passengers are dropped off 

def top_dropoff(trip_year,counter):
        
    top_drop_off_spots=pd.pivot_table(data=chicago_3_year_raw_data,values='taxi_id',index=[('drop_off_lat_lon')],columns=['trip_year'],aggfunc='count',fill_value=0,margins=2,margins_name="total")
    top_drop_off_spots=top_drop_off_spots.loc[:,trip_year]
    top_drop_off_spots=pd.DataFrame(top_drop_off_spots)
    top_drop_off_spots=top_drop_off_spots.drop('total').sort_values(by=trip_year,ascending=False).head(counter)
    top_drop_off_spots=top_drop_off_spots.reset_index()
    top_drop_off_spots[['lat','lon']]=top_drop_off_spots['drop_off_lat_lon'].apply(pd.Series)
    top_drop_off_spots_1=pd.DataFrame(top_drop_off_spots.drop('drop_off_lat_lon',axis=1))
    top_drop_off_spots_1=top_drop_off_spots.iloc[:,1:]

    top_drop_off_spots_1['address'] = top_drop_off_spots_1.apply(lambda row: geolocator.reverse((row['lat'], row['lon'])), axis=1)
    
    top_drop_off_spots_1['lat']=pd.DataFrame.convert_objects(top_drop_off_spots_1['lat'],convert_numeric=True)
    top_drop_off_spots_1['lon']=pd.DataFrame.convert_objects(top_drop_off_spots_1['lon'],convert_numeric=True)
    top_drop_off_spots_1['address']=top_drop_off_spots_1['address'].astype(str)
    
    lat=top_drop_off_spots_1['lat']
    lon=top_drop_off_spots_1['lon']
    
    lat_mean = top_drop_off_spots_1['lat'].mean()
    lon_mean = top_drop_off_spots_1['lon'].mean()
    
    locationlist = top_drop_off_spots_1[["lat","lon"]].values.tolist()
    labels = top_drop_off_spots_1["address"].values.tolist()

    chimap = folium.Map([lat_mean,lon_mean], zoom_start=13)
    folium.TileLayer('cartodbpositron').add_to(chimap)
    
    drop_offs_size=top_drop_off_spots_1.iloc[:,0]
        
    for i in range(len(locationlist)):
        popup = folium.Popup(str(labels[i]) +" -> (total drop offs  = "+ str(int(drop_offs_size[i])) + ")", parse_html=True)
        icon=folium.Icon(color='red')
        folium.CircleMarker(locationlist[i], popup=popup, icon = icon,color='red',fill_color='red',radius = drop_offs_size[i]/1000,fill=True).add_to(chimap)

    return(chimap)
    
top_dropoff('2015',10)

