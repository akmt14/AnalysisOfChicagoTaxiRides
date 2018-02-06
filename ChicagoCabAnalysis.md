
#### Chicago Cab Analysis

I had to come up with some real world data to analyze for my final Data Mining project. I had read about Uber & Lyft driving the regular cab drivers out of business & this intrigued me. Chicago has the second largest cab fleets & according to "rumors" they were facing extinction. According to reports, approximately 40% of the 7000 strong cab fleet had not had a fare for the month of March 2017 & I wanted to see this for myself.

I came across the Chicago cab data through the Chicago Data Portal. Their data was from years 2013-2016 & was of size 40 GB & had around 127 million records. I extracted the data using an API into my Python notebook. Unfortunately, due to system restrictions, I was able to work on only a small sample set of this data (~1%). I analyzed this sample data & my findings resonated with what some articles mentioned about how cab drivers are facing unemployment.



```python
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
```

    WARNING:root:Requests made without an app_token will be subject to strict throttling limits.
    

The complete dataset is available at the [Chicago Data Portal](https://data.cityofchicago.org/Transportation/Taxi-Trips/wrvz-psew/data) has more data than what my system could handle. It didn't make sense to download a 40 GB csv file, read it & then analyze it. Hence I accessed the data through the Socrata Open Data API (SODA). System restrictions meant that I could not access the entire dataset. Hence I decided to subset the data. I took a sample of each of the 3 years for sizes 100k, 250k & 500k, structured & aggregated the data & plotted them.


```python
sample_100k_2016 = client.get("wrvz-psew",limit=100000,
                              where="trip_start_timestamp > '2016-01-01T00:00:00' \
                              AND trip_start_timestamp <= '2016-12-31T00:00:00' AND \
                              pickup_census_tract IS NOT NULL AND \
                              dropoff_census_tract IS NOT NULL AND \
                              company IS NOT NULL")

```


```python
sample_250k_2016 = client.get("wrvz-psew",limit=250000,
                              where="trip_start_timestamp > '2016-01-01T00:00:00' \
                              AND trip_start_timestamp <= '2016-12-31T00:00:00' AND \
                              pickup_census_tract IS NOT NULL AND \
                              dropoff_census_tract IS NOT NULL AND \
                              company IS NOT NULL")

```


```python
sample_500k_2016 = client.get("wrvz-psew",limit=500000,
                              where="trip_start_timestamp > '2016-01-01T00:00:00' \
                              AND trip_start_timestamp <= '2016-12-31T00:00:00' AND \
                              pickup_census_tract IS NOT NULL AND \
                              dropoff_census_tract IS NOT NULL AND \
                              company IS NOT NULL")

```


```python
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

```


```python
# comparing samples of different sizes for 2016

sample_data_grouped=pd.pivot_table(chicago_appended_sample_data,values='taxi_id',index=('trip_month'),columns=('sample'),fill_value="",margins=2,margins_name="total",aggfunc=[len])
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
sample_data_grouped=sample_data_grouped.reindex(months)
sample_data_grouped = sample_data_grouped.xs('len', axis=1, drop_level=True)
sample_data_grouped
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>sample</th>
      <th>sample_100k</th>
      <th>sample_250k</th>
      <th>sample_500k</th>
      <th>total</th>
    </tr>
    <tr>
      <th>trip_month</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Jan</th>
      <td>8019.0</td>
      <td>20215.0</td>
      <td>40450.0</td>
      <td>68684.0</td>
    </tr>
    <tr>
      <th>Feb</th>
      <td>8565.0</td>
      <td>21273.0</td>
      <td>42557.0</td>
      <td>72395.0</td>
    </tr>
    <tr>
      <th>Mar</th>
      <td>10038.0</td>
      <td>24888.0</td>
      <td>49311.0</td>
      <td>84237.0</td>
    </tr>
    <tr>
      <th>Apr</th>
      <td>9862.0</td>
      <td>24855.0</td>
      <td>49635.0</td>
      <td>84352.0</td>
    </tr>
    <tr>
      <th>May</th>
      <td>9905.0</td>
      <td>24666.0</td>
      <td>49524.0</td>
      <td>84095.0</td>
    </tr>
    <tr>
      <th>Jun</th>
      <td>9925.0</td>
      <td>24827.0</td>
      <td>49470.0</td>
      <td>84222.0</td>
    </tr>
    <tr>
      <th>Jul</th>
      <td>8723.0</td>
      <td>21790.0</td>
      <td>44000.0</td>
      <td>74513.0</td>
    </tr>
    <tr>
      <th>Aug</th>
      <td>7920.0</td>
      <td>19876.0</td>
      <td>39781.0</td>
      <td>67577.0</td>
    </tr>
    <tr>
      <th>Sep</th>
      <td>7191.0</td>
      <td>18475.0</td>
      <td>36372.0</td>
      <td>62038.0</td>
    </tr>
    <tr>
      <th>Oct</th>
      <td>7594.0</td>
      <td>18916.0</td>
      <td>38447.0</td>
      <td>64957.0</td>
    </tr>
    <tr>
      <th>Nov</th>
      <td>6502.0</td>
      <td>15944.0</td>
      <td>31782.0</td>
      <td>54228.0</td>
    </tr>
    <tr>
      <th>Dec</th>
      <td>5756.0</td>
      <td>14275.0</td>
      <td>28671.0</td>
      <td>48702.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
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
```


![png](output_8_0.png)


The above graph consists of 3 sample sets of 100k, 250k & 500k for the year 2016. Looking at the samples, it seems rather obvious that the samples follow a certain trend & that they can be considered to be an approximate but similar representation of the population.
I opted for the middle value of 250k & proceeded with my analysis for the years 2014-16.


```python
chicago_cab_data_raw_2016 = client.get("wrvz-psew",limit=250000,where="trip_start_timestamp > '2016-01-01T00:00:00' AND trip_start_timestamp <= '2016-12-31T00:00:00' AND dropoff_centroid_longitude IS NOT NULL AND dropoff_centroid_latitude IS NOT NULL AND pickup_centroid_longitude IS NOT NULL AND pickup_centroid_latitude IS NOT NULL AND company IS NOT NULL")
chicago_cab_data_raw_2015 = client.get("wrvz-psew",limit=250000,where="trip_start_timestamp > '2015-01-01T00:00:00' AND trip_start_timestamp <= '2015-12-31T00:00:00' AND dropoff_centroid_longitude IS NOT NULL AND dropoff_centroid_latitude IS NOT NULL AND pickup_centroid_longitude IS NOT NULL AND pickup_centroid_latitude IS NOT NULL AND company IS NOT NULL")
chicago_cab_data_raw_2014 = client.get("wrvz-psew",limit=250000,where="trip_start_timestamp > '2014-01-01T00:00:00' AND trip_start_timestamp <= '2014-12-31T00:00:00' AND dropoff_centroid_longitude IS NOT NULL AND dropoff_centroid_latitude IS NOT NULL AND pickup_centroid_longitude IS NOT NULL AND pickup_centroid_latitude IS NOT NULL AND company IS NOT NULL")

chicago_cab_data_2016 = pd.DataFrame.from_records(chicago_cab_data_raw_2016)
chicago_cab_data_2015 = pd.DataFrame.from_records(chicago_cab_data_raw_2015)
chicago_cab_data_2014 = pd.DataFrame.from_records(chicago_cab_data_raw_2014)
```


```python
chicago_3_year_raw_data = pd.concat([chicago_cab_data_2014,chicago_cab_data_2015,chicago_cab_data_2016])
chicago_3_year_raw_data.head(2)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>company</th>
      <th>dropoff_census_tract</th>
      <th>dropoff_centroid_latitude</th>
      <th>dropoff_centroid_location</th>
      <th>dropoff_centroid_longitude</th>
      <th>dropoff_community_area</th>
      <th>extras</th>
      <th>fare</th>
      <th>payment_type</th>
      <th>pickup_census_tract</th>
      <th>...</th>
      <th>pickup_community_area</th>
      <th>taxi_id</th>
      <th>tips</th>
      <th>tolls</th>
      <th>trip_end_timestamp</th>
      <th>trip_id</th>
      <th>trip_miles</th>
      <th>trip_seconds</th>
      <th>trip_start_timestamp</th>
      <th>trip_total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Taxi Affiliation Services</td>
      <td>NaN</td>
      <td>41.899602111</td>
      <td>{'type': 'Point', 'coordinates': [-87.63330803...</td>
      <td>-87.633308037</td>
      <td>8</td>
      <td>0</td>
      <td>7.25</td>
      <td>Cash</td>
      <td>NaN</td>
      <td>...</td>
      <td>32</td>
      <td>c75a6874181b5e7410d6e250ea8ce2ade4fdd068a9729f...</td>
      <td>0</td>
      <td>0</td>
      <td>2014-02-05T05:00:00.000</td>
      <td>0ed7d6ce715e69d8ca485c47d45d84c054384b12</td>
      <td>0</td>
      <td>600</td>
      <td>2014-02-05T04:45:00.000</td>
      <td>7.25</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Dispatch Taxi Affiliation</td>
      <td>17031320100</td>
      <td>41.884987192</td>
      <td>{'type': 'Point', 'coordinates': [-87.62099291...</td>
      <td>-87.620992913</td>
      <td>32</td>
      <td>0</td>
      <td>5.25</td>
      <td>Cash</td>
      <td>17031839100</td>
      <td>...</td>
      <td>32</td>
      <td>d41f06822eb03048a1c433a0e5f3c69e7d75672b13db5b...</td>
      <td>0</td>
      <td>0</td>
      <td>2014-09-19T16:45:00.000</td>
      <td>0ed7d8d0446b6c539eef5d7714a80db392947ec5</td>
      <td>0</td>
      <td>360</td>
      <td>2014-09-19T16:45:00.000</td>
      <td>5.25</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 23 columns</p>
</div>




```python
chicago_3_year_raw_data=chicago_3_year_raw_data.iloc[:,[14,18,0,10,12,11,13,2,4,3,5,21,17,19,20,7,6,15,16,22,8]]
chicago_3_year_raw_data["trip_date"]=pd.to_datetime(chicago_3_year_raw_data["trip_start_timestamp"])
chicago_3_year_raw_data.head(2)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>taxi_id</th>
      <th>trip_id</th>
      <th>company</th>
      <th>pickup_centroid_latitude</th>
      <th>pickup_centroid_longitude</th>
      <th>pickup_centroid_location</th>
      <th>pickup_community_area</th>
      <th>dropoff_centroid_latitude</th>
      <th>dropoff_centroid_longitude</th>
      <th>dropoff_centroid_location</th>
      <th>...</th>
      <th>trip_end_timestamp</th>
      <th>trip_miles</th>
      <th>trip_seconds</th>
      <th>fare</th>
      <th>extras</th>
      <th>tips</th>
      <th>tolls</th>
      <th>trip_total</th>
      <th>payment_type</th>
      <th>trip_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>c75a6874181b5e7410d6e250ea8ce2ade4fdd068a9729f...</td>
      <td>0ed7d6ce715e69d8ca485c47d45d84c054384b12</td>
      <td>Taxi Affiliation Services</td>
      <td>41.878865584</td>
      <td>-87.625192142</td>
      <td>{'type': 'Point', 'coordinates': [-87.62519214...</td>
      <td>32</td>
      <td>41.899602111</td>
      <td>-87.633308037</td>
      <td>{'type': 'Point', 'coordinates': [-87.63330803...</td>
      <td>...</td>
      <td>2014-02-05T05:00:00.000</td>
      <td>0</td>
      <td>600</td>
      <td>7.25</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7.25</td>
      <td>Cash</td>
      <td>2014-02-05 04:45:00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>d41f06822eb03048a1c433a0e5f3c69e7d75672b13db5b...</td>
      <td>0ed7d8d0446b6c539eef5d7714a80db392947ec5</td>
      <td>Dispatch Taxi Affiliation</td>
      <td>41.880994471</td>
      <td>-87.632746489</td>
      <td>{'type': 'Point', 'coordinates': [-87.63274648...</td>
      <td>32</td>
      <td>41.884987192</td>
      <td>-87.620992913</td>
      <td>{'type': 'Point', 'coordinates': [-87.62099291...</td>
      <td>...</td>
      <td>2014-09-19T16:45:00.000</td>
      <td>0</td>
      <td>360</td>
      <td>5.25</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5.25</td>
      <td>Cash</td>
      <td>2014-09-19 16:45:00</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 22 columns</p>
</div>




```python
# manipulating trip_date column to extract day & time 

chicago_3_year_raw_data["trip_date"]=chicago_3_year_raw_data["trip_date"].astype(str)
chicago_3_year_raw_data["trip_start_time"]=chicago_3_year_raw_data["trip_date"].str[-8:]
chicago_3_year_raw_data["trip_date"]=chicago_3_year_raw_data["trip_date"].str[:10]
chicago_3_year_raw_data["trip_year"]=chicago_3_year_raw_data["trip_date"].str[:4]
chicago_3_year_raw_data["trip_month"]=chicago_3_year_raw_data["trip_date"]

chicago_3_year_raw_data.head(2)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>taxi_id</th>
      <th>trip_id</th>
      <th>company</th>
      <th>pickup_centroid_latitude</th>
      <th>pickup_centroid_longitude</th>
      <th>pickup_centroid_location</th>
      <th>pickup_community_area</th>
      <th>dropoff_centroid_latitude</th>
      <th>dropoff_centroid_longitude</th>
      <th>dropoff_centroid_location</th>
      <th>...</th>
      <th>fare</th>
      <th>extras</th>
      <th>tips</th>
      <th>tolls</th>
      <th>trip_total</th>
      <th>payment_type</th>
      <th>trip_date</th>
      <th>trip_start_time</th>
      <th>trip_year</th>
      <th>trip_month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>c75a6874181b5e7410d6e250ea8ce2ade4fdd068a9729f...</td>
      <td>0ed7d6ce715e69d8ca485c47d45d84c054384b12</td>
      <td>Taxi Affiliation Services</td>
      <td>41.878865584</td>
      <td>-87.625192142</td>
      <td>{'type': 'Point', 'coordinates': [-87.62519214...</td>
      <td>32</td>
      <td>41.899602111</td>
      <td>-87.633308037</td>
      <td>{'type': 'Point', 'coordinates': [-87.63330803...</td>
      <td>...</td>
      <td>7.25</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7.25</td>
      <td>Cash</td>
      <td>2014-02-05</td>
      <td>04:45:00</td>
      <td>2014</td>
      <td>2014-02-05</td>
    </tr>
    <tr>
      <th>1</th>
      <td>d41f06822eb03048a1c433a0e5f3c69e7d75672b13db5b...</td>
      <td>0ed7d8d0446b6c539eef5d7714a80db392947ec5</td>
      <td>Dispatch Taxi Affiliation</td>
      <td>41.880994471</td>
      <td>-87.632746489</td>
      <td>{'type': 'Point', 'coordinates': [-87.63274648...</td>
      <td>32</td>
      <td>41.884987192</td>
      <td>-87.620992913</td>
      <td>{'type': 'Point', 'coordinates': [-87.62099291...</td>
      <td>...</td>
      <td>5.25</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5.25</td>
      <td>Cash</td>
      <td>2014-09-19</td>
      <td>16:45:00</td>
      <td>2014</td>
      <td>2014-09-19</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 25 columns</p>
</div>




```python
# getting month name from month number

chicago_3_year_raw_data["trip_month"]=pd.to_datetime(chicago_3_year_raw_data["trip_month"])
chicago_3_year_raw_data["trip_month"]=chicago_3_year_raw_data["trip_month"].apply(lambda x: datetime.datetime.strftime(x,'%b'))
chicago_3_year_raw_data.tail(2)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>taxi_id</th>
      <th>trip_id</th>
      <th>company</th>
      <th>pickup_centroid_latitude</th>
      <th>pickup_centroid_longitude</th>
      <th>pickup_centroid_location</th>
      <th>pickup_community_area</th>
      <th>dropoff_centroid_latitude</th>
      <th>dropoff_centroid_longitude</th>
      <th>dropoff_centroid_location</th>
      <th>...</th>
      <th>fare</th>
      <th>extras</th>
      <th>tips</th>
      <th>tolls</th>
      <th>trip_total</th>
      <th>payment_type</th>
      <th>trip_date</th>
      <th>trip_start_time</th>
      <th>trip_year</th>
      <th>trip_month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>249998</th>
      <td>9d63e251e254a38306995b5f4401beef8aa295292f49e2...</td>
      <td>136398b20b233cd6b7ef2adab90aaec094f45641</td>
      <td>Blue Ribbon Taxi Association Inc.</td>
      <td>41.859349715</td>
      <td>-87.617358006</td>
      <td>{'type': 'Point', 'coordinates': [-87.61735800...</td>
      <td>33</td>
      <td>41.884987192</td>
      <td>-87.620992913</td>
      <td>{'type': 'Point', 'coordinates': [-87.62099291...</td>
      <td>...</td>
      <td>11</td>
      <td>0</td>
      <td>2.22</td>
      <td>0</td>
      <td>13.22</td>
      <td>Credit Card</td>
      <td>2016-09-15</td>
      <td>16:15:00</td>
      <td>2016</td>
      <td>Sep</td>
    </tr>
    <tr>
      <th>249999</th>
      <td>cb0238f1280ec16111baa2e62030dc943d6ca4119acdb2...</td>
      <td>136398fcab39e898897a0d0db56c4ced3340dbfd</td>
      <td>Taxi Affiliation Services</td>
      <td>41.914616286</td>
      <td>-87.631717366</td>
      <td>{'type': 'Point', 'coordinates': [-87.63171736...</td>
      <td>7</td>
      <td>41.90749193</td>
      <td>-87.63576009</td>
      <td>{'type': 'Point', 'coordinates': [-87.63576009...</td>
      <td>...</td>
      <td>6.5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6.5</td>
      <td>Cash</td>
      <td>2016-04-04</td>
      <td>16:45:00</td>
      <td>2016</td>
      <td>Apr</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 25 columns</p>
</div>




```python
# month on month taxi ride counts across years

chicago_3_year_structured=pd.pivot_table(chicago_3_year_raw_data,values='taxi_id',index=('trip_month'),columns=('trip_year'),fill_value="",margins=2,margins_name="total",aggfunc=[len])
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
chicago_3_year_structured=chicago_3_year_structured.reindex(months)
chicago_3_year_structured

```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="4" halign="left">len</th>
    </tr>
    <tr>
      <th>trip_year</th>
      <th>2014</th>
      <th>2015</th>
      <th>2016</th>
      <th>total</th>
    </tr>
    <tr>
      <th>trip_month</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Jan</th>
      <td>18584.0</td>
      <td>20403.0</td>
      <td>21146.0</td>
      <td>60133.0</td>
    </tr>
    <tr>
      <th>Feb</th>
      <td>19306.0</td>
      <td>20571.0</td>
      <td>21696.0</td>
      <td>61573.0</td>
    </tr>
    <tr>
      <th>Mar</th>
      <td>22053.0</td>
      <td>23072.0</td>
      <td>24568.0</td>
      <td>69693.0</td>
    </tr>
    <tr>
      <th>Apr</th>
      <td>21565.0</td>
      <td>21898.0</td>
      <td>24031.0</td>
      <td>67494.0</td>
    </tr>
    <tr>
      <th>May</th>
      <td>23456.0</td>
      <td>23766.0</td>
      <td>24337.0</td>
      <td>71559.0</td>
    </tr>
    <tr>
      <th>Jun</th>
      <td>22879.0</td>
      <td>22365.0</td>
      <td>24195.0</td>
      <td>69439.0</td>
    </tr>
    <tr>
      <th>Jul</th>
      <td>21495.0</td>
      <td>21810.0</td>
      <td>21752.0</td>
      <td>65057.0</td>
    </tr>
    <tr>
      <th>Aug</th>
      <td>21490.0</td>
      <td>20608.0</td>
      <td>20287.0</td>
      <td>62385.0</td>
    </tr>
    <tr>
      <th>Sep</th>
      <td>20136.0</td>
      <td>19320.0</td>
      <td>18763.0</td>
      <td>58219.0</td>
    </tr>
    <tr>
      <th>Oct</th>
      <td>20985.0</td>
      <td>20850.0</td>
      <td>18860.0</td>
      <td>60695.0</td>
    </tr>
    <tr>
      <th>Nov</th>
      <td>19495.0</td>
      <td>17927.0</td>
      <td>15894.0</td>
      <td>53316.0</td>
    </tr>
    <tr>
      <th>Dec</th>
      <td>18556.0</td>
      <td>17410.0</td>
      <td>14471.0</td>
      <td>50437.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# dropping level for ease

chicago_3_year_structured = chicago_3_year_structured.xs('len', axis=1, drop_level=True)
chicago_3_year_structured
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>trip_year</th>
      <th>2014</th>
      <th>2015</th>
      <th>2016</th>
      <th>total</th>
    </tr>
    <tr>
      <th>trip_month</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Jan</th>
      <td>18584.0</td>
      <td>20403.0</td>
      <td>21146.0</td>
      <td>60133.0</td>
    </tr>
    <tr>
      <th>Feb</th>
      <td>19306.0</td>
      <td>20571.0</td>
      <td>21696.0</td>
      <td>61573.0</td>
    </tr>
    <tr>
      <th>Mar</th>
      <td>22053.0</td>
      <td>23072.0</td>
      <td>24568.0</td>
      <td>69693.0</td>
    </tr>
    <tr>
      <th>Apr</th>
      <td>21565.0</td>
      <td>21898.0</td>
      <td>24031.0</td>
      <td>67494.0</td>
    </tr>
    <tr>
      <th>May</th>
      <td>23456.0</td>
      <td>23766.0</td>
      <td>24337.0</td>
      <td>71559.0</td>
    </tr>
    <tr>
      <th>Jun</th>
      <td>22879.0</td>
      <td>22365.0</td>
      <td>24195.0</td>
      <td>69439.0</td>
    </tr>
    <tr>
      <th>Jul</th>
      <td>21495.0</td>
      <td>21810.0</td>
      <td>21752.0</td>
      <td>65057.0</td>
    </tr>
    <tr>
      <th>Aug</th>
      <td>21490.0</td>
      <td>20608.0</td>
      <td>20287.0</td>
      <td>62385.0</td>
    </tr>
    <tr>
      <th>Sep</th>
      <td>20136.0</td>
      <td>19320.0</td>
      <td>18763.0</td>
      <td>58219.0</td>
    </tr>
    <tr>
      <th>Oct</th>
      <td>20985.0</td>
      <td>20850.0</td>
      <td>18860.0</td>
      <td>60695.0</td>
    </tr>
    <tr>
      <th>Nov</th>
      <td>19495.0</td>
      <td>17927.0</td>
      <td>15894.0</td>
      <td>53316.0</td>
    </tr>
    <tr>
      <th>Dec</th>
      <td>18556.0</td>
      <td>17410.0</td>
      <td>14471.0</td>
      <td>50437.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
chicago_3_year_structured = pd.DataFrame(chicago_3_year_structured.iloc[0:12,0:3])
chicago_3_year_structured
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>trip_year</th>
      <th>2014</th>
      <th>2015</th>
      <th>2016</th>
    </tr>
    <tr>
      <th>trip_month</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Jan</th>
      <td>18584.0</td>
      <td>20403.0</td>
      <td>21146.0</td>
    </tr>
    <tr>
      <th>Feb</th>
      <td>19306.0</td>
      <td>20571.0</td>
      <td>21696.0</td>
    </tr>
    <tr>
      <th>Mar</th>
      <td>22053.0</td>
      <td>23072.0</td>
      <td>24568.0</td>
    </tr>
    <tr>
      <th>Apr</th>
      <td>21565.0</td>
      <td>21898.0</td>
      <td>24031.0</td>
    </tr>
    <tr>
      <th>May</th>
      <td>23456.0</td>
      <td>23766.0</td>
      <td>24337.0</td>
    </tr>
    <tr>
      <th>Jun</th>
      <td>22879.0</td>
      <td>22365.0</td>
      <td>24195.0</td>
    </tr>
    <tr>
      <th>Jul</th>
      <td>21495.0</td>
      <td>21810.0</td>
      <td>21752.0</td>
    </tr>
    <tr>
      <th>Aug</th>
      <td>21490.0</td>
      <td>20608.0</td>
      <td>20287.0</td>
    </tr>
    <tr>
      <th>Sep</th>
      <td>20136.0</td>
      <td>19320.0</td>
      <td>18763.0</td>
    </tr>
    <tr>
      <th>Oct</th>
      <td>20985.0</td>
      <td>20850.0</td>
      <td>18860.0</td>
    </tr>
    <tr>
      <th>Nov</th>
      <td>19495.0</td>
      <td>17927.0</td>
      <td>15894.0</td>
    </tr>
    <tr>
      <th>Dec</th>
      <td>18556.0</td>
      <td>17410.0</td>
      <td>14471.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
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
```


![png](output_18_0.png)


As we can see, the number of rides go up in each of the 3 years as each year begins. However, as the second half of the year begins, the ride numbers go down drastically. 

Are cab drivers really losing business due to Uber & lyft?


```python
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

```

                active taxis
    year_month              
    2014 - Jan          2479
    2014 - Feb          2525
    2014 - Mar          2587
    2014 - Apr          2693
    2014 - May          2689
    2014 - Jun          2633
    2014 - Jul          2655
    2014 - Aug          2648
    2014 - Sep          2582
    2014 - Oct          2581
    2014 - Nov          2575
    2014 - Dec          2556
    2015 - Jan          2479
    2015 - Feb          2501
    2015 - Mar          2567
    2015 - Apr          2574
    2015 - May          2614
    2015 - Jun          2601
    2015 - Jul          2597
    2015 - Aug          2527
    2015 - Sep          2485
    2015 - Oct          2536
    2015 - Nov          2517
    2015 - Dec          2445
    2016 - Jan          2292
    2016 - Feb          2313
    2016 - Mar          2345
    2016 - Apr          2370
    2016 - May          2334
    2016 - Jun          2315
    2016 - Jul          2240
    2016 - Aug          2175
    2016 - Sep          2103
    2016 - Oct          1978
    2016 - Nov          1880
    2016 - Dec          1750
    


![png](output_20_1.png)


As we can see, the unique count of taxi id has gone down from ~2500 to ~1800 over a period of 3 years. Which is almost a 70% decrease - which resonates with the below mentioned article. [Number of Chicago taxi drivers hits 10-year low as ride-share companies take off - Chicago Tribune (Dec 16, 2017)](http://www.chicagotribune.com/news/ct-chicago-taxi-driver-decline-met-20161214-story.html)

I then wanted to look at the daily ride trend across the entire timeline hoping to find some interesting trends.


```python
# day on day total rides across 2014-16

day_on_day_trend_2016 = chicago_3_year_raw_data[(chicago_3_year_raw_data["trip_date"] >= '2014-01-01') & (chicago_3_year_raw_data["trip_date"] <= '2016-12-31')]
day_on_day_trend_2016 = day_on_day_trend_2016.rename(columns={'trip_id':'count of trips'})
ax = pd.pivot_table(day_on_day_trend_2016, values='count of trips', columns=['trip_date'], fill_value="",aggfunc=[len]).T
ax = ax.xs('len', axis=0, drop_level=True)
```


```python
ax.plot(figsize=(20,6),color='red')
pp.xlabel('daily ride trend',color='black')
pp.xticks(color = 'black')
pp.yticks(color = 'black')
pp.ylabel('total rides',color='black')
pp.title('ride trend across the year',color='black')
pp.xticks(rotation='vertical')
pp.legend(loc='center left', bbox_to_anchor=(1,.97))
pp.show()
```


![png](output_23_0.png)


As we can see in the above image, there are distinct peaks & troughs across the entire timeline. However, for each year, there is a distinct peak at around March. To get a better idea of the data, I aggregated the data on "total rides on a per day basis", sorted the data & looked at the top 10 days with most number of cab rides. Also, using the date, I extracted the days corresponding to these dates.


```python
# top 10 days with most cab rides (2014-16)

chicago_3_year_raw_data['count']=1
reqd_cols=chicago_3_year_raw_data.pivot_table(columns="trip_date",aggfunc=sum).T.reset_index().sort_values(by='count',ascending=False)
reqd_cols['trip_date']=reqd_cols['trip_date'].astype('datetime64[ns]')
reqd_cols['day']=reqd_cols['trip_date'].dt.strftime('%A')
reqd_cols['year']=reqd_cols['trip_date'].dt.strftime('%Y')
reqd_cols.head(10)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>trip_date</th>
      <th>count</th>
      <th>day</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>437</th>
      <td>2015-03-14</td>
      <td>1208</td>
      <td>Saturday</td>
      <td>2015</td>
    </tr>
    <tr>
      <th>801</th>
      <td>2016-03-12</td>
      <td>1196</td>
      <td>Saturday</td>
      <td>2016</td>
    </tr>
    <tr>
      <th>73</th>
      <td>2014-03-15</td>
      <td>1190</td>
      <td>Saturday</td>
      <td>2014</td>
    </tr>
    <tr>
      <th>884</th>
      <td>2016-06-03</td>
      <td>1040</td>
      <td>Friday</td>
      <td>2016</td>
    </tr>
    <tr>
      <th>514</th>
      <td>2015-05-30</td>
      <td>989</td>
      <td>Saturday</td>
      <td>2015</td>
    </tr>
    <tr>
      <th>135</th>
      <td>2014-05-16</td>
      <td>974</td>
      <td>Friday</td>
      <td>2014</td>
    </tr>
    <tr>
      <th>786</th>
      <td>2016-02-26</td>
      <td>973</td>
      <td>Friday</td>
      <td>2016</td>
    </tr>
    <tr>
      <th>136</th>
      <td>2014-05-17</td>
      <td>971</td>
      <td>Saturday</td>
      <td>2014</td>
    </tr>
    <tr>
      <th>150</th>
      <td>2014-05-31</td>
      <td>967</td>
      <td>Saturday</td>
      <td>2014</td>
    </tr>
    <tr>
      <th>821</th>
      <td>2016-04-01</td>
      <td>963</td>
      <td>Friday</td>
      <td>2016</td>
    </tr>
  </tbody>
</table>
</div>



One interesting observation was that in the top ten days with most cabs hired, each and every day was either on a Friday or a Saturday. So one hypothesis that I could come up with is that the weekend, the Chicagoans prefer to take cabs over the weekend more than the weekday. However, this would need more clarity. So, I plotted the cab hire counts across the entire week for each of the 3 years. 


```python
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
```


![png](output_27_0.png)


As evident from the graph above, my hypothesis held true. Weekday rides are low as compared to the weekend & as the weekend approaches, the ride count keeps increasing.

I then wanted to see if the there has been a significant change in average ride costs or average ride times.


```python
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
```

    D:\Anaconda\lib\site-packages\pandas\core\frame.py:2450: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      self[k1] = value[k2]
    




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="3" halign="left">avg ride cost (in USD)</th>
    </tr>
    <tr>
      <th>trip_year</th>
      <th>2014</th>
      <th>2015</th>
      <th>2016</th>
    </tr>
    <tr>
      <th>trip_month</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Jan</th>
      <td>11.81</td>
      <td>12.59</td>
      <td>13.95</td>
    </tr>
    <tr>
      <th>Feb</th>
      <td>12.58</td>
      <td>12.52</td>
      <td>14.64</td>
    </tr>
    <tr>
      <th>Mar</th>
      <td>12.29</td>
      <td>13.47</td>
      <td>14.66</td>
    </tr>
    <tr>
      <th>Apr</th>
      <td>12.74</td>
      <td>13.62</td>
      <td>15.11</td>
    </tr>
    <tr>
      <th>May</th>
      <td>13.27</td>
      <td>13.49</td>
      <td>15.79</td>
    </tr>
    <tr>
      <th>Jun</th>
      <td>13.59</td>
      <td>13.95</td>
      <td>15.70</td>
    </tr>
    <tr>
      <th>Jul</th>
      <td>13.30</td>
      <td>13.59</td>
      <td>15.25</td>
    </tr>
    <tr>
      <th>Aug</th>
      <td>12.90</td>
      <td>13.88</td>
      <td>15.43</td>
    </tr>
    <tr>
      <th>Sep</th>
      <td>13.61</td>
      <td>14.31</td>
      <td>15.75</td>
    </tr>
    <tr>
      <th>Oct</th>
      <td>13.60</td>
      <td>13.87</td>
      <td>16.09</td>
    </tr>
    <tr>
      <th>Nov</th>
      <td>13.08</td>
      <td>13.96</td>
      <td>15.48</td>
    </tr>
    <tr>
      <th>Dec</th>
      <td>12.78</td>
      <td>12.55</td>
      <td>13.74</td>
    </tr>
  </tbody>
</table>
</div>




```python
chicago_3_year_structured_money_1 = chicago_3_year_structured_money.xs('avg ride cost (in USD)', axis=1, drop_level=True)
chicago_3_year_structured_money_1 = chicago_3_year_structured_money_1.unstack()
```


```python
chicago_3_year_structured_money_1.plot(figsize=(15,5),color='red')

pp.xlabel('average cost of cab rides (2014-16)',color='black')
pp.xticks(color = 'black')
pp.yticks(color = 'black')
pp.ylabel('average cost of rides (in USD)',color='black')
pp.title('average trip cost across the years',color='black')
pp.xticks(rotation='vertical')

pp.show()
```


![png](output_31_0.png)


As we can see, the average ride cost at the start of the year is slightly less than $ 12. However, just before 2016 ended, the average ride cost was around $ 16 before reducogm to around $ 12. 


```python
# average month on month trip duration in minutes (2014-16)

chicago_3_year_structured_duration=pd.pivot_table(chicago_3_year_raw_data1,values=('trip_seconds'),index=('trip_month'),columns=('trip_year'),margins=2,margins_name="total",aggfunc=[np.average])
chicago_3_year_structured_duration=(chicago_3_year_structured_duration/60).round(2)
chicago_3_year_structured_duration=chicago_3_year_structured_duration.iloc[:12,:3]
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
chicago_3_year_structured_duration=chicago_3_year_structured_duration.reindex(months)
chicago_3_year_structured_duration.columns = chicago_3_year_structured_duration.columns.set_levels(['avg trip time (in minutes)'], level=0)
chicago_3_year_structured_duration

```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="3" halign="left">avg trip time (in minutes)</th>
    </tr>
    <tr>
      <th>trip_year</th>
      <th>2014</th>
      <th>2015</th>
      <th>2016</th>
    </tr>
    <tr>
      <th>trip_month</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Jan</th>
      <td>11.40</td>
      <td>11.44</td>
      <td>11.32</td>
    </tr>
    <tr>
      <th>Feb</th>
      <td>11.70</td>
      <td>11.97</td>
      <td>11.69</td>
    </tr>
    <tr>
      <th>Mar</th>
      <td>11.89</td>
      <td>12.28</td>
      <td>12.22</td>
    </tr>
    <tr>
      <th>Apr</th>
      <td>12.07</td>
      <td>12.61</td>
      <td>12.78</td>
    </tr>
    <tr>
      <th>May</th>
      <td>12.85</td>
      <td>13.25</td>
      <td>13.32</td>
    </tr>
    <tr>
      <th>Jun</th>
      <td>12.92</td>
      <td>13.65</td>
      <td>14.14</td>
    </tr>
    <tr>
      <th>Jul</th>
      <td>12.77</td>
      <td>13.48</td>
      <td>13.35</td>
    </tr>
    <tr>
      <th>Aug</th>
      <td>12.53</td>
      <td>12.99</td>
      <td>13.17</td>
    </tr>
    <tr>
      <th>Sep</th>
      <td>12.88</td>
      <td>13.59</td>
      <td>13.71</td>
    </tr>
    <tr>
      <th>Oct</th>
      <td>12.93</td>
      <td>13.46</td>
      <td>13.71</td>
    </tr>
    <tr>
      <th>Nov</th>
      <td>12.47</td>
      <td>12.87</td>
      <td>13.22</td>
    </tr>
    <tr>
      <th>Dec</th>
      <td>11.87</td>
      <td>12.08</td>
      <td>12.17</td>
    </tr>
  </tbody>
</table>
</div>




```python
chicago_3_year_structured_duration_1 = chicago_3_year_structured_duration.xs('avg trip time (in minutes)', axis=1,drop_level=True)
chicago_3_year_structured_duration_1 =chicago_3_year_structured_duration_1.unstack()
```


```python
chicago_3_year_structured_duration_1.plot(figsize=(15,5),color='green')

pp.xlabel('average duration of cab rides (2014-16)',color='black')
pp.xticks(color = 'black')
pp.yticks(color = 'black')
pp.ylabel('average duration of rides (in minutes)',color='black')
pp.title('average trip duration across the years',color='black')
pp.xticks(rotation='vertical')

pp.show()
```


![png](output_35_0.png)


As we can see, the average trip duration at the start of the year is around 11.5 minutes. However, just before 2016 ended, the average ride cost was around 13 minutes.

The above 2 data points do not make much sense when looked at individually. So, I merged these 2 datasets & plotted the data overlapped data points.


```python
# average month on month across years ride cost vs trip duration

compare_trip_cost_and_duration = [chicago_3_year_structured_money, chicago_3_year_structured_duration]
compare_trip_cost_and_duration_1 = pd.concat(compare_trip_cost_and_duration, axis=1)
compare_trip_cost_and_duration_1
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="3" halign="left">avg ride cost (in USD)</th>
      <th colspan="3" halign="left">avg trip time (in minutes)</th>
    </tr>
    <tr>
      <th>trip_year</th>
      <th>2014</th>
      <th>2015</th>
      <th>2016</th>
      <th>2014</th>
      <th>2015</th>
      <th>2016</th>
    </tr>
    <tr>
      <th>trip_month</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Jan</th>
      <td>11.81</td>
      <td>12.59</td>
      <td>13.95</td>
      <td>11.40</td>
      <td>11.44</td>
      <td>11.32</td>
    </tr>
    <tr>
      <th>Feb</th>
      <td>12.58</td>
      <td>12.52</td>
      <td>14.64</td>
      <td>11.70</td>
      <td>11.97</td>
      <td>11.69</td>
    </tr>
    <tr>
      <th>Mar</th>
      <td>12.29</td>
      <td>13.47</td>
      <td>14.66</td>
      <td>11.89</td>
      <td>12.28</td>
      <td>12.22</td>
    </tr>
    <tr>
      <th>Apr</th>
      <td>12.74</td>
      <td>13.62</td>
      <td>15.11</td>
      <td>12.07</td>
      <td>12.61</td>
      <td>12.78</td>
    </tr>
    <tr>
      <th>May</th>
      <td>13.27</td>
      <td>13.49</td>
      <td>15.79</td>
      <td>12.85</td>
      <td>13.25</td>
      <td>13.32</td>
    </tr>
    <tr>
      <th>Jun</th>
      <td>13.59</td>
      <td>13.95</td>
      <td>15.70</td>
      <td>12.92</td>
      <td>13.65</td>
      <td>14.14</td>
    </tr>
    <tr>
      <th>Jul</th>
      <td>13.30</td>
      <td>13.59</td>
      <td>15.25</td>
      <td>12.77</td>
      <td>13.48</td>
      <td>13.35</td>
    </tr>
    <tr>
      <th>Aug</th>
      <td>12.90</td>
      <td>13.88</td>
      <td>15.43</td>
      <td>12.53</td>
      <td>12.99</td>
      <td>13.17</td>
    </tr>
    <tr>
      <th>Sep</th>
      <td>13.61</td>
      <td>14.31</td>
      <td>15.75</td>
      <td>12.88</td>
      <td>13.59</td>
      <td>13.71</td>
    </tr>
    <tr>
      <th>Oct</th>
      <td>13.60</td>
      <td>13.87</td>
      <td>16.09</td>
      <td>12.93</td>
      <td>13.46</td>
      <td>13.71</td>
    </tr>
    <tr>
      <th>Nov</th>
      <td>13.08</td>
      <td>13.96</td>
      <td>15.48</td>
      <td>12.47</td>
      <td>12.87</td>
      <td>13.22</td>
    </tr>
    <tr>
      <th>Dec</th>
      <td>12.78</td>
      <td>12.55</td>
      <td>13.74</td>
      <td>11.87</td>
      <td>12.08</td>
      <td>12.17</td>
    </tr>
  </tbody>
</table>
</div>




```python
# restructuring data for plotting purposes

compare_trip_cost_and_duration_1=compare_trip_cost_and_duration_1.stack()
compare_trip_cost_and_duration_1=compare_trip_cost_and_duration_1.reset_index()

compare_trip_cost_and_duration_1['year_month'] = compare_trip_cost_and_duration_1["trip_year"] +" - "+compare_trip_cost_and_duration_1["trip_month"]
compare_trip_cost_and_duration_1=compare_trip_cost_and_duration_1.iloc[:,2:]

compare_trip_cost_and_duration_1=compare_trip_cost_and_duration_1.set_index('year_month')
compare_trip_cost_and_duration_1.head(5)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>avg ride cost (in USD)</th>
      <th>avg trip time (in minutes)</th>
    </tr>
    <tr>
      <th>year_month</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2014 - Jan</th>
      <td>11.81</td>
      <td>11.40</td>
    </tr>
    <tr>
      <th>2015 - Jan</th>
      <td>12.59</td>
      <td>11.44</td>
    </tr>
    <tr>
      <th>2016 - Jan</th>
      <td>13.95</td>
      <td>11.32</td>
    </tr>
    <tr>
      <th>2014 - Feb</th>
      <td>12.58</td>
      <td>11.70</td>
    </tr>
    <tr>
      <th>2015 - Feb</th>
      <td>12.52</td>
      <td>11.97</td>
    </tr>
  </tbody>
</table>
</div>




```python
year_month=['2014 - Jan','2014 - Feb','2014 - Mar','2014 - Apr','2014 - May','2014 - Jun','2014 - Jul','2014 - Aug','2014 - Sep','2014 - Oct','2014 - Nov','2014 - Dec','2015 - Jan','2015 - Feb','2015 - Mar','2015 - Apr','2015 - May','2015 - Jun','2015 - Jul','2015 - Aug','2015 - Sep','2015 - Oct','2015 - Nov','2015 - Dec','2016 - Jan','2016 - Feb','2016 - Mar','2016 - Apr','2016 - May','2016 - Jun','2016 - Jul','2016 - Aug','2016 - Sep','2016 - Oct','2016 - Nov','2016 - Dec']
compare_trip_cost_and_duration_1=compare_trip_cost_and_duration_1.reindex(year_month)
compare_trip_cost_and_duration_1.head(5)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>avg ride cost (in USD)</th>
      <th>avg trip time (in minutes)</th>
    </tr>
    <tr>
      <th>year_month</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2014 - Jan</th>
      <td>11.81</td>
      <td>11.40</td>
    </tr>
    <tr>
      <th>2014 - Feb</th>
      <td>12.58</td>
      <td>11.70</td>
    </tr>
    <tr>
      <th>2014 - Mar</th>
      <td>12.29</td>
      <td>11.89</td>
    </tr>
    <tr>
      <th>2014 - Apr</th>
      <td>12.74</td>
      <td>12.07</td>
    </tr>
    <tr>
      <th>2014 - May</th>
      <td>13.27</td>
      <td>12.85</td>
    </tr>
  </tbody>
</table>
</div>




```python
compare_trip_cost_and_duration_1.plot(figsize=(15,5),color=['RED','GREEN'])

pp.xlabel('month on month - (2014-16)',color='black')
pp.xticks(color = 'black')
pp.yticks(color = 'black')
pp.title('avg cost of cab rides v/s avg trip duration ',color='black')
pp.xticks(rotation='vertical')

pp.show()
```


![png](output_40_0.png)


As we can see initially, the 2 variables, average cab ride times & average cost of taxi hires go hand in hand. However, as the year 2016 begins, even though average ride time has remained a more or less a consistent value, the average cab ride cost has increased tremendously!  

Then to check if there was any particluar cab ride fare method that was more common than others, I analysed the payment type field of this dataset.


```python
# credit card payments vs cash payments

chicago_3_year_raw_data
type_of_payment = chicago_3_year_raw_data[["trip_id","payment_type","trip_year","trip_month"]]
type_of_payment_grouped = type_of_payment.pivot_table(index="trip_month",columns=["trip_year","payment_type"],values="trip_id",aggfunc=np.count_nonzero,margins=0) 

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
type_of_payment_grouped=type_of_payment_grouped.reindex(months)
type_of_payment_grouped=type_of_payment_grouped.drop(['Dispute','No Charge','Unknown'],axis=1,level=1)
```


```python
type_of_payment_grouped
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th>trip_year</th>
      <th colspan="2" halign="left">2014</th>
      <th colspan="2" halign="left">2015</th>
      <th colspan="2" halign="left">2016</th>
    </tr>
    <tr>
      <th>payment_type</th>
      <th>Cash</th>
      <th>Credit Card</th>
      <th>Cash</th>
      <th>Credit Card</th>
      <th>Cash</th>
      <th>Credit Card</th>
    </tr>
    <tr>
      <th>trip_month</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Jan</th>
      <td>13406</td>
      <td>4960</td>
      <td>12919</td>
      <td>7275</td>
      <td>12090</td>
      <td>8834</td>
    </tr>
    <tr>
      <th>Feb</th>
      <td>13542</td>
      <td>5550</td>
      <td>12726</td>
      <td>7601</td>
      <td>11873</td>
      <td>9555</td>
    </tr>
    <tr>
      <th>Mar</th>
      <td>15571</td>
      <td>6277</td>
      <td>13936</td>
      <td>8876</td>
      <td>13497</td>
      <td>10782</td>
    </tr>
    <tr>
      <th>Apr</th>
      <td>14711</td>
      <td>6637</td>
      <td>12908</td>
      <td>8743</td>
      <td>12876</td>
      <td>10876</td>
    </tr>
    <tr>
      <th>May</th>
      <td>15912</td>
      <td>7332</td>
      <td>14129</td>
      <td>9389</td>
      <td>13094</td>
      <td>10963</td>
    </tr>
    <tr>
      <th>Jun</th>
      <td>15256</td>
      <td>7430</td>
      <td>13026</td>
      <td>9108</td>
      <td>12975</td>
      <td>10956</td>
    </tr>
    <tr>
      <th>Jul</th>
      <td>14452</td>
      <td>6872</td>
      <td>13428</td>
      <td>8159</td>
      <td>12347</td>
      <td>9202</td>
    </tr>
    <tr>
      <th>Aug</th>
      <td>14529</td>
      <td>6789</td>
      <td>12468</td>
      <td>7936</td>
      <td>11330</td>
      <td>8745</td>
    </tr>
    <tr>
      <th>Sep</th>
      <td>12895</td>
      <td>7085</td>
      <td>11296</td>
      <td>7817</td>
      <td>10215</td>
      <td>8323</td>
    </tr>
    <tr>
      <th>Oct</th>
      <td>13337</td>
      <td>7475</td>
      <td>11965</td>
      <td>8668</td>
      <td>9939</td>
      <td>8700</td>
    </tr>
    <tr>
      <th>Nov</th>
      <td>12389</td>
      <td>6904</td>
      <td>10126</td>
      <td>7626</td>
      <td>8405</td>
      <td>7246</td>
    </tr>
    <tr>
      <th>Dec</th>
      <td>12229</td>
      <td>6162</td>
      <td>10262</td>
      <td>6964</td>
      <td>8166</td>
      <td>6118</td>
    </tr>
  </tbody>
</table>
</div>




```python
type_of_payment_percent = round(100*type_of_payment_grouped.div(type_of_payment_grouped.sum(axis=1, level=0), level=0),1)
type_of_payment_percent_1 = type_of_payment_percent.astype(str)+"%"
type_of_payment_percent_1
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th>trip_year</th>
      <th colspan="2" halign="left">2014</th>
      <th colspan="2" halign="left">2015</th>
      <th colspan="2" halign="left">2016</th>
    </tr>
    <tr>
      <th>payment_type</th>
      <th>Cash</th>
      <th>Credit Card</th>
      <th>Cash</th>
      <th>Credit Card</th>
      <th>Cash</th>
      <th>Credit Card</th>
    </tr>
    <tr>
      <th>trip_month</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Jan</th>
      <td>73.0%</td>
      <td>27.0%</td>
      <td>64.0%</td>
      <td>36.0%</td>
      <td>57.8%</td>
      <td>42.2%</td>
    </tr>
    <tr>
      <th>Feb</th>
      <td>70.9%</td>
      <td>29.1%</td>
      <td>62.6%</td>
      <td>37.4%</td>
      <td>55.4%</td>
      <td>44.6%</td>
    </tr>
    <tr>
      <th>Mar</th>
      <td>71.3%</td>
      <td>28.7%</td>
      <td>61.1%</td>
      <td>38.9%</td>
      <td>55.6%</td>
      <td>44.4%</td>
    </tr>
    <tr>
      <th>Apr</th>
      <td>68.9%</td>
      <td>31.1%</td>
      <td>59.6%</td>
      <td>40.4%</td>
      <td>54.2%</td>
      <td>45.8%</td>
    </tr>
    <tr>
      <th>May</th>
      <td>68.5%</td>
      <td>31.5%</td>
      <td>60.1%</td>
      <td>39.9%</td>
      <td>54.4%</td>
      <td>45.6%</td>
    </tr>
    <tr>
      <th>Jun</th>
      <td>67.2%</td>
      <td>32.8%</td>
      <td>58.9%</td>
      <td>41.1%</td>
      <td>54.2%</td>
      <td>45.8%</td>
    </tr>
    <tr>
      <th>Jul</th>
      <td>67.8%</td>
      <td>32.2%</td>
      <td>62.2%</td>
      <td>37.8%</td>
      <td>57.3%</td>
      <td>42.7%</td>
    </tr>
    <tr>
      <th>Aug</th>
      <td>68.2%</td>
      <td>31.8%</td>
      <td>61.1%</td>
      <td>38.9%</td>
      <td>56.4%</td>
      <td>43.6%</td>
    </tr>
    <tr>
      <th>Sep</th>
      <td>64.5%</td>
      <td>35.5%</td>
      <td>59.1%</td>
      <td>40.9%</td>
      <td>55.1%</td>
      <td>44.9%</td>
    </tr>
    <tr>
      <th>Oct</th>
      <td>64.1%</td>
      <td>35.9%</td>
      <td>58.0%</td>
      <td>42.0%</td>
      <td>53.3%</td>
      <td>46.7%</td>
    </tr>
    <tr>
      <th>Nov</th>
      <td>64.2%</td>
      <td>35.8%</td>
      <td>57.0%</td>
      <td>43.0%</td>
      <td>53.7%</td>
      <td>46.3%</td>
    </tr>
    <tr>
      <th>Dec</th>
      <td>66.5%</td>
      <td>33.5%</td>
      <td>59.6%</td>
      <td>40.4%</td>
      <td>57.2%</td>
      <td>42.8%</td>
    </tr>
  </tbody>
</table>
</div>




```python
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
```


![png](output_45_0.png)


At the start of 2014, Cash payments contributed to approximately 73%. However, over time of 3 years, we can see that payment through cash reduced to approximately 56% & payment through Credit card increased in the same time period.

I wanted to have a look at the most common pickup/drop-off spots for cab drivers. In order to do that I had to manipulate the latitudes and longitudes into formats that would be acceptable by the package geopy. I passed the latitude, longitude values through the geopy function & reverse geolocated the addresses of the pickup/ drop-offs on a map using the folium package. Basically, I created a function that enables the user to look up any number of top pickup/drop off places in Chicago. The size of the bubble indicates the count of the number of pickup/ drop offs.


```python
# manipulating data for plotting pick up & drop offs on a map

chicago_3_year_raw_data['pick_up_lat_lon']=chicago_3_year_raw_data[['pickup_centroid_latitude','pickup_centroid_longitude']].apply(tuple,axis=1)
chicago_3_year_raw_data['drop_off_lat_lon']=chicago_3_year_raw_data[['dropoff_centroid_latitude','dropoff_centroid_longitude']].apply(tuple,axis=1)
chicago_3_year_raw_data.dtypes
```




    taxi_id                       object
    trip_id                       object
    company                       object
    pickup_centroid_latitude      object
    pickup_centroid_longitude     object
    pickup_centroid_location      object
    pickup_community_area         object
    dropoff_centroid_latitude     object
    dropoff_centroid_longitude    object
    dropoff_centroid_location     object
    dropoff_community_area        object
    trip_start_timestamp          object
    trip_end_timestamp            object
    trip_miles                    object
    trip_seconds                  object
    fare                          object
    extras                        object
    tips                          object
    tolls                         object
    trip_total                    object
    payment_type                  object
    trip_date                     object
    trip_start_time               object
    trip_year                     object
    trip_month                    object
    count                          int64
    pick_up_lat_lon               object
    drop_off_lat_lon              object
    dtype: object




```python
top_drop_off_spots=pd.crosstab(index=(chicago_3_year_raw_data['drop_off_lat_lon']),columns='count')
top_drop_off_spots=top_drop_off_spots.sort_values(ascending=False,by='count').head(10)
top_drop_off_spots=top_drop_off_spots.reset_index()
top_drop_off_spots[['lat','lon']]=top_drop_off_spots['drop_off_lat_lon'].apply(pd.Series)

# formatting lat & lon for drop offs

top_pick_up_spots=pd.crosstab(index=(chicago_3_year_raw_data['pick_up_lat_lon']),columns='count')
top_pick_up_spots=top_pick_up_spots.sort_values(ascending=False,by='count').head(10)
top_pick_up_spots=top_pick_up_spots.reset_index()
top_pick_up_spots[['lat','lon']]=top_pick_up_spots['pick_up_lat_lon'].apply(pd.Series)

```


```python
from geopy.geocoders import Nominatim
geolocator = Nominatim()
```


```python
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

```

    D:\Anaconda\lib\site-packages\ipykernel_launcher.py:16: FutureWarning: convert_objects is deprecated.  Use the data-type specific converters pd.to_datetime, pd.to_timedelta and pd.to_numeric.
      app.launch_new_instance()
    D:\Anaconda\lib\site-packages\ipykernel_launcher.py:17: FutureWarning: convert_objects is deprecated.  Use the data-type specific converters pd.to_datetime, pd.to_timedelta and pd.to_numeric.
    




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><iframe src="data:text/html;charset=utf-8;base64,PCFET0NUWVBFIGh0bWw+CjxoZWFkPiAgICAKICAgIDxtZXRhIGh0dHAtZXF1aXY9ImNvbnRlbnQtdHlwZSIgY29udGVudD0idGV4dC9odG1sOyBjaGFyc2V0PVVURi04IiAvPgogICAgPHNjcmlwdD5MX1BSRUZFUl9DQU5WQVMgPSBmYWxzZTsgTF9OT19UT1VDSCA9IGZhbHNlOyBMX0RJU0FCTEVfM0QgPSBmYWxzZTs8L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS4yLjAvZGlzdC9sZWFmbGV0LmpzIj48L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2FqYXguZ29vZ2xlYXBpcy5jb20vYWpheC9saWJzL2pxdWVyeS8xLjExLjEvanF1ZXJ5Lm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvanMvYm9vdHN0cmFwLm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9jZG5qcy5jbG91ZGZsYXJlLmNvbS9hamF4L2xpYnMvTGVhZmxldC5hd2Vzb21lLW1hcmtlcnMvMi4wLjIvbGVhZmxldC5hd2Vzb21lLW1hcmtlcnMuanMiPjwvc2NyaXB0PgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS4yLjAvZGlzdC9sZWFmbGV0LmNzcyIgLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvY3NzL2Jvb3RzdHJhcC5taW4uY3NzIiAvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL21heGNkbi5ib290c3RyYXBjZG4uY29tL2Jvb3RzdHJhcC8zLjIuMC9jc3MvYm9vdHN0cmFwLXRoZW1lLm1pbi5jc3MiIC8+CiAgICA8bGluayByZWw9InN0eWxlc2hlZXQiIGhyZWY9Imh0dHBzOi8vbWF4Y2RuLmJvb3RzdHJhcGNkbi5jb20vZm9udC1hd2Vzb21lLzQuNi4zL2Nzcy9mb250LWF3ZXNvbWUubWluLmNzcyIgLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9jZG5qcy5jbG91ZGZsYXJlLmNvbS9hamF4L2xpYnMvTGVhZmxldC5hd2Vzb21lLW1hcmtlcnMvMi4wLjIvbGVhZmxldC5hd2Vzb21lLW1hcmtlcnMuY3NzIiAvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL3Jhd2dpdC5jb20vcHl0aG9uLXZpc3VhbGl6YXRpb24vZm9saXVtL21hc3Rlci9mb2xpdW0vdGVtcGxhdGVzL2xlYWZsZXQuYXdlc29tZS5yb3RhdGUuY3NzIiAvPgogICAgPHN0eWxlPmh0bWwsIGJvZHkge3dpZHRoOiAxMDAlO2hlaWdodDogMTAwJTttYXJnaW46IDA7cGFkZGluZzogMDt9PC9zdHlsZT4KICAgIDxzdHlsZT4jbWFwIHtwb3NpdGlvbjphYnNvbHV0ZTt0b3A6MDtib3R0b206MDtyaWdodDowO2xlZnQ6MDt9PC9zdHlsZT4KICAgIAogICAgICAgICAgICA8c3R5bGU+ICNtYXBfNzA5YjcyOGY3NjVlNDE1NDhkOWFkMmU1NTgwNGQyMjMgewogICAgICAgICAgICAgICAgcG9zaXRpb24gOiByZWxhdGl2ZTsKICAgICAgICAgICAgICAgIHdpZHRoIDogMTAwLjAlOwogICAgICAgICAgICAgICAgaGVpZ2h0OiAxMDAuMCU7CiAgICAgICAgICAgICAgICBsZWZ0OiAwLjAlOwogICAgICAgICAgICAgICAgdG9wOiAwLjAlOwogICAgICAgICAgICAgICAgfQogICAgICAgICAgICA8L3N0eWxlPgogICAgICAgIAo8L2hlYWQ+Cjxib2R5PiAgICAKICAgIAogICAgICAgICAgICA8ZGl2IGNsYXNzPSJmb2xpdW0tbWFwIiBpZD0ibWFwXzcwOWI3MjhmNzY1ZTQxNTQ4ZDlhZDJlNTU4MDRkMjIzIiA+PC9kaXY+CiAgICAgICAgCjwvYm9keT4KPHNjcmlwdD4gICAgCiAgICAKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGJvdW5kcyA9IG51bGw7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgdmFyIG1hcF83MDliNzI4Zjc2NWU0MTU0OGQ5YWQyZTU1ODA0ZDIyMyA9IEwubWFwKAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgJ21hcF83MDliNzI4Zjc2NWU0MTU0OGQ5YWQyZTU1ODA0ZDIyMycsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB7Y2VudGVyOiBbNDEuOTAzNjgyNDU4MiwtODcuNjYwMzUyNTcwMV0sCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB6b29tOiAxMiwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIG1heEJvdW5kczogYm91bmRzLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgbGF5ZXJzOiBbXSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHdvcmxkQ29weUp1bXA6IGZhbHNlLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgY3JzOiBMLkNSUy5FUFNHMzg1NwogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB9KTsKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHRpbGVfbGF5ZXJfNjc2YWFlOWQ4NjllNDVlYjkyYTFjMTU2NzZmMzAxNWIgPSBMLnRpbGVMYXllcigKICAgICAgICAgICAgICAgICdodHRwczovL3tzfS50aWxlLm9wZW5zdHJlZXRtYXAub3JnL3t6fS97eH0ve3l9LnBuZycsCiAgICAgICAgICAgICAgICB7CiAgImF0dHJpYnV0aW9uIjogbnVsbCwKICAiZGV0ZWN0UmV0aW5hIjogZmFsc2UsCiAgIm1heFpvb20iOiAxOCwKICAibWluWm9vbSI6IDEsCiAgIm5vV3JhcCI6IGZhbHNlLAogICJzdWJkb21haW5zIjogImFiYyIKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNzA5YjcyOGY3NjVlNDE1NDhkOWFkMmU1NTgwNGQyMjMpOwogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciB0aWxlX2xheWVyX2RjMzNjMWU3ZDRiMzRlMDdhNzJhNDEwZTk5NmI0ZDFiID0gTC50aWxlTGF5ZXIoCiAgICAgICAgICAgICAgICAnaHR0cHM6Ly9jYXJ0b2RiLWJhc2VtYXBzLXtzfS5nbG9iYWwuc3NsLmZhc3RseS5uZXQvbGlnaHRfYWxsL3t6fS97eH0ve3l9LnBuZycsCiAgICAgICAgICAgICAgICB7CiAgImF0dHJpYnV0aW9uIjogbnVsbCwKICAiZGV0ZWN0UmV0aW5hIjogZmFsc2UsCiAgIm1heFpvb20iOiAxOCwKICAibWluWm9vbSI6IDEsCiAgIm5vV3JhcCI6IGZhbHNlLAogICJzdWJkb21haW5zIjogImFiYyIKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNzA5YjcyOGY3NjVlNDE1NDhkOWFkMmU1NTgwNGQyMjMpOwogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzczNTA1Yjk5OGMzNTQ4MmU4YjI4NDVjYTg5YTlmZmI2ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDEuODgwOTk0NDcxLC04Ny42MzI3NDY0ODldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiZ3JlZW4iLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICJncmVlbiIsCiAgImZpbGxPcGFjaXR5IjogMC4yLAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogMjYuNTEsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNzA5YjcyOGY3NjVlNDE1NDhkOWFkMmU1NTgwNGQyMjMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfOGMxNTllZTZhNjk1NDQ5NmFhZTU4MDBiNzVmOGZmNWUgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZjVkODIyY2Y0OWUwNDA4OGFhN2E1NTY3N2IzNTVjZmMgPSAkKCc8ZGl2IGlkPSJodG1sX2Y1ZDgyMmNmNDllMDQwODhhYTdhNTU2NzdiMzU1Y2ZjIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Ob3J0aGVybiBUcnVzdCBCYW5rLCA1MCwgU291dGggTGEgU2FsbGUgU3RyZWV0LCBMb29wLCBDaGljYWdvLCBDb29rIENvdW50eSwgSWxsaW5vaXMsIDYwNjAzLCBVbml0ZWQgU3RhdGVzIG9mIEFtZXJpY2EgLSZndDsgKHRvdGFsIHBpY2sgdXBzICA9IDI2NTEwKTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfOGMxNTllZTZhNjk1NDQ5NmFhZTU4MDBiNzVmOGZmNWUuc2V0Q29udGVudChodG1sX2Y1ZDgyMmNmNDllMDQwODhhYTdhNTU2NzdiMzU1Y2ZjKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzczNTA1Yjk5OGMzNTQ4MmU4YjI4NDVjYTg5YTlmZmI2LmJpbmRQb3B1cChwb3B1cF84YzE1OWVlNmE2OTU0NDk2YWFlNTgwMGI3NWY4ZmY1ZSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl84MzdjZGYxZWEwNDI0MzczYmMwOGNkMDdlMmMwMzUzZSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQxLjg4NDk4NzE5MiwtODcuNjIwOTkyOTEzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImdyZWVuIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiZ3JlZW4iLAogICJmaWxsT3BhY2l0eSI6IDAuMiwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDE1LjI2LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzcwOWI3MjhmNzY1ZTQxNTQ4ZDlhZDJlNTU4MDRkMjIzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzU2MDhhN2RkZGNjNDRiZjU4MmI0YjA2MDVjNDM5ZTA4ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2E2YWEzMzBhNDRkMDRkNjg5Y2FkMjc5MWQwNWFhMDFkID0gJCgnPGRpdiBpZD0iaHRtbF9hNmFhMzMwYTQ0ZDA0ZDY4OWNhZDI3OTFkMDVhYTAxZCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QW9uIENlbnRlciwgMjAwLCBFYXN0IFJhbmRvbHBoIFN0cmVldCwgTG9vcCwgQ2hpY2FnbywgQ29vayBDb3VudHksIElsbGlub2lzLCA2MDYxMSwgVW5pdGVkIFN0YXRlcyBvZiBBbWVyaWNhIC0mZ3Q7ICh0b3RhbCBwaWNrIHVwcyAgPSAxNTI2MCk8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzU2MDhhN2RkZGNjNDRiZjU4MmI0YjA2MDVjNDM5ZTA4LnNldENvbnRlbnQoaHRtbF9hNmFhMzMwYTQ0ZDA0ZDY4OWNhZDI3OTFkMDVhYTAxZCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl84MzdjZGYxZWEwNDI0MzczYmMwOGNkMDdlMmMwMzUzZS5iaW5kUG9wdXAocG9wdXBfNTYwOGE3ZGRkY2M0NGJmNTgyYjRiMDYwNWM0MzllMDgpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZTA0MjUzMzMxYjgwNDkzYzhmMDA2ZmJiOWU0N2FmYzAgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0MS44OTI1MDc3ODEsLTg3LjYyNjIxNDkwNl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJncmVlbiIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogImdyZWVuIiwKICAiZmlsbE9wYWNpdHkiOiAwLjIsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiAxMS45NDcsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNzA5YjcyOGY3NjVlNDE1NDhkOWFkMmU1NTgwNGQyMjMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZWFkODRjZTIzZTY4NGE3M2FmOTM3MDUyMzQwODUwZjMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNzJiNzhjNWE5NWZlNDQ5YTg1NDczNmU3MDlkYjJhZTkgPSAkKCc8ZGl2IGlkPSJodG1sXzcyYjc4YzVhOTVmZTQ0OWE4NTQ3MzZlNzA5ZGIyYWU5IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5SdXNoLU9oaW8tV2FiYXNoIFNlbGYgUGFyaywgNTAsIEVhc3QgT2hpbyBTdHJlZXQsIE1hZ25pZmljZW50IE1pbGUsIENoaWNhZ28sIENvb2sgQ291bnR5LCBJbGxpbm9pcywgNjA2MTEsIFVuaXRlZCBTdGF0ZXMgb2YgQW1lcmljYSAtJmd0OyAodG90YWwgcGljayB1cHMgID0gMTE5NDcpPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9lYWQ4NGNlMjNlNjg0YTczYWY5MzcwNTIzNDA4NTBmMy5zZXRDb250ZW50KGh0bWxfNzJiNzhjNWE5NWZlNDQ5YTg1NDczNmU3MDlkYjJhZTkpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZTA0MjUzMzMxYjgwNDkzYzhmMDA2ZmJiOWU0N2FmYzAuYmluZFBvcHVwKHBvcHVwX2VhZDg0Y2UyM2U2ODRhNzNhZjkzNzA1MjM0MDg1MGYzKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzAxNTYzNTU5Yjc4NTQyZjI5NjE2Mzg1NjEzODlmMzExID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDEuODk5NjAyMTExLC04Ny42MzMzMDgwMzddLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiZ3JlZW4iLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICJncmVlbiIsCiAgImZpbGxPcGFjaXR5IjogMC4yLAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogMTEuMTk1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzcwOWI3MjhmNzY1ZTQxNTQ4ZDlhZDJlNTU4MDRkMjIzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzM4MzAzMzc0ZDU5NjRiYjdiOWJhMzI2NGUzOGU1MjczID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzQ0NzkzZjA0YTkyZTQzNDc5YTBkZTFiMmE1NTZiMzE1ID0gJCgnPGRpdiBpZD0iaHRtbF80NDc5M2YwNGE5MmU0MzQ3OWEwZGUxYjJhNTU2YjMxNSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RHJ5ZXIgSGFsbCAtIE1vb2R5IEJpYmxlIEluc3RpdHV0ZSwgOTMwLCBOb3J0aCBMYSBTYWxsZSBCb3VsZXZhcmQsIE5lYXIgTm9ydGggU2lkZSwgQ2hpY2FnbywgQ29vayBDb3VudHksIElsbGlub2lzLCA2MDYxMSwgVW5pdGVkIFN0YXRlcyBvZiBBbWVyaWNhIC0mZ3Q7ICh0b3RhbCBwaWNrIHVwcyAgPSAxMTE5NSk8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzM4MzAzMzc0ZDU5NjRiYjdiOWJhMzI2NGUzOGU1MjczLnNldENvbnRlbnQoaHRtbF80NDc5M2YwNGE5MmU0MzQ3OWEwZGUxYjJhNTU2YjMxNSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8wMTU2MzU1OWI3ODU0MmYyOTYxNjM4NTYxMzg5ZjMxMS5iaW5kUG9wdXAocG9wdXBfMzgzMDMzNzRkNTk2NGJiN2I5YmEzMjY0ZTM4ZTUyNzMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYzNhZDU5MmY4NTJlNDcxMmIyM2Q3Zjg3ZjU1NGJmOTggPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0MS44OTIwNDIxMzYsLTg3LjYzMTg2Mzk1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImdyZWVuIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiZ3JlZW4iLAogICJmaWxsT3BhY2l0eSI6IDAuMiwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDExLjEzOSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF83MDliNzI4Zjc2NWU0MTU0OGQ5YWQyZTU1ODA0ZDIyMyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF80Yzg5YTlmNWI5Y2U0MjJjYjQyYWUzOTMzY2I0NTFmZSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF80NmFkODFlYWU4OGE0NmZiYTM5NWI2NWYzOWI3ZDg2MCA9ICQoJzxkaXYgaWQ9Imh0bWxfNDZhZDgxZWFlODhhNDZmYmEzOTViNjVmMzliN2Q4NjAiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkJlc3QgV2VzdGVybiBQbHVzIFJpdmVyIE5vcnRoLCAxMjUsIFdlc3QgT2hpbyBTdHJlZXQsIE1hZ25pZmljZW50IE1pbGUsIENoaWNhZ28sIENvb2sgQ291bnR5LCBJbGxpbm9pcywgNjA2MTEsIFVuaXRlZCBTdGF0ZXMgb2YgQW1lcmljYSAtJmd0OyAodG90YWwgcGljayB1cHMgID0gMTExMzkpPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF80Yzg5YTlmNWI5Y2U0MjJjYjQyYWUzOTMzY2I0NTFmZS5zZXRDb250ZW50KGh0bWxfNDZhZDgxZWFlODhhNDZmYmEzOTViNjVmMzliN2Q4NjApOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYzNhZDU5MmY4NTJlNDcxMmIyM2Q3Zjg3ZjU1NGJmOTguYmluZFBvcHVwKHBvcHVwXzRjODlhOWY1YjljZTQyMmNiNDJhZTM5MzNjYjQ1MWZlKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzQ2Y2YwOGFhYmE4ZTQ2YTBhMzQxMDMxMWFjMGQxNjc2ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDEuOTc5MDcwODIsLTg3LjkwMzAzOTY2MTAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImdyZWVuIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiZ3JlZW4iLAogICJmaWxsT3BhY2l0eSI6IDAuMiwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDEwLjI2NCwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF83MDliNzI4Zjc2NWU0MTU0OGQ5YWQyZTU1ODA0ZDIyMyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8wMjcwODIyNzUxZDA0YjdkODg0ZWViZTBmMzlhY2I4MCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF83NGNjOTlmMDdiMjI0ZjhkOTVhOGExNDhkZWY1NzQ0OSA9ICQoJzxkaXYgaWQ9Imh0bWxfNzRjYzk5ZjA3YjIyNGY4ZDk1YThhMTQ4ZGVmNTc0NDkiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkJ1cyBEaXNwYXRjaCwgTyYjMzk7SGFyZSBBcnJpdmFscywgTyYjMzk7SGFyZSwgQ2hpY2FnbywgQ29vayBDb3VudHksIElsbGlub2lzLCA2MDAxOCwgVW5pdGVkIFN0YXRlcyBvZiBBbWVyaWNhIC0mZ3Q7ICh0b3RhbCBwaWNrIHVwcyAgPSAxMDI2NCk8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzAyNzA4MjI3NTFkMDRiN2Q4ODRlZWJlMGYzOWFjYjgwLnNldENvbnRlbnQoaHRtbF83NGNjOTlmMDdiMjI0ZjhkOTVhOGExNDhkZWY1NzQ0OSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl80NmNmMDhhYWJhOGU0NmEwYTM0MTAzMTFhYzBkMTY3Ni5iaW5kUG9wdXAocG9wdXBfMDI3MDgyMjc1MWQwNGI3ZDg4NGVlYmUwZjM5YWNiODApOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYTZkMWRmM2M5MjlhNDYwZDhlZDU4YzUwYWQ2ZjhhNWIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0MS44NzkyNTUwODQsLTg3LjY0MjY0ODk5OF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJncmVlbiIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogImdyZWVuIiwKICAiZmlsbE9wYWNpdHkiOiAwLjIsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA5LjE2NCwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF83MDliNzI4Zjc2NWU0MTU0OGQ5YWQyZTU1ODA0ZDIyMyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF81OTk3MGY3NWNiN2Y0NzhiOTFkOTNmN2ViZjdjYjM5NiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF83OWMyMDc1NWEyMzI0ZGMzYWRmNjZiNDdkMzI3YmU2ZCA9ICQoJzxkaXYgaWQ9Imh0bWxfNzljMjA3NTVhMjMyNGRjM2FkZjY2YjQ3ZDMyN2JlNmQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkFsJiMzOTtzICMxIEl0YWxpYW4gQmVlZiwgNjAxLCBXZXN0IEFkYW1zIFN0cmVldCwgR3JlZWt0b3duLCBDaGljYWdvLCBDb29rIENvdW50eSwgSWxsaW5vaXMsIDYwNjYxLCBVbml0ZWQgU3RhdGVzIG9mIEFtZXJpY2EgLSZndDsgKHRvdGFsIHBpY2sgdXBzICA9IDkxNjQpPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF81OTk3MGY3NWNiN2Y0NzhiOTFkOTNmN2ViZjdjYjM5Ni5zZXRDb250ZW50KGh0bWxfNzljMjA3NTVhMjMyNGRjM2FkZjY2YjQ3ZDMyN2JlNmQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYTZkMWRmM2M5MjlhNDYwZDhlZDU4YzUwYWQ2ZjhhNWIuYmluZFBvcHVwKHBvcHVwXzU5OTcwZjc1Y2I3ZjQ3OGI5MWQ5M2Y3ZWJmN2NiMzk2KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2QwODZkMWNmMDhiZjQ2YzNhZGViNWFjZDk4ZmU0ZDFiID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDEuOTQ0MjI2NjAxLC04Ny42NTU5OTgxODJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiZ3JlZW4iLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICJncmVlbiIsCiAgImZpbGxPcGFjaXR5IjogMC4yLAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogOS4xMDgsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNzA5YjcyOGY3NjVlNDE1NDhkOWFkMmU1NTgwNGQyMjMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfOWQxYTdjNDZiM2E0NDE2N2JmOTBlODliZDc1OTIwYmEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMzVlZGY0MTM2ZjE3NDRiOGI4MzQwMzE5NzdjMzcxNTYgPSAkKCc8ZGl2IGlkPSJodG1sXzM1ZWRmNDEzNmYxNzQ0YjhiODM0MDMxOTc3YzM3MTU2IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij4xMDQxLTEwNDMsIFdlc3QgTmV3cG9ydCBBdmVudWUsIFdyaWdsZXl2aWxsZSwgVXB0b3duLCBDaGljYWdvLCBDb29rIENvdW50eSwgSWxsaW5vaXMsIDYwNjU3LCBVbml0ZWQgU3RhdGVzIG9mIEFtZXJpY2EgLSZndDsgKHRvdGFsIHBpY2sgdXBzICA9IDkxMDgpPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF85ZDFhN2M0NmIzYTQ0MTY3YmY5MGU4OWJkNzU5MjBiYS5zZXRDb250ZW50KGh0bWxfMzVlZGY0MTM2ZjE3NDRiOGI4MzQwMzE5NzdjMzcxNTYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZDA4NmQxY2YwOGJmNDZjM2FkZWI1YWNkOThmZTRkMWIuYmluZFBvcHVwKHBvcHVwXzlkMWE3YzQ2YjNhNDQxNjdiZjkwZTg5YmQ3NTkyMGJhKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2FmYzFiMDllZjkxMjQzMWQ5Yzg2MzYxYzVkZGJhOTlhID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDEuODkwOTIyMDI2LC04Ny42MTg4NjgzNTVdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiZ3JlZW4iLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICJncmVlbiIsCiAgImZpbGxPcGFjaXR5IjogMC4yLAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNy4zNDIsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNzA5YjcyOGY3NjVlNDE1NDhkOWFkMmU1NTgwNGQyMjMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfODY4OThmM2NjMDRiNDBjYWExZTNhYzM3NWZmMGZkYzUgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfY2M4YjY1ZmJlYzI3NGQxMjkyNjE3ZGU1OTVmOTc5MTMgPSAkKCc8ZGl2IGlkPSJodG1sX2NjOGI2NWZiZWMyNzRkMTI5MjYxN2RlNTk1Zjk3OTEzIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij40NjUsIE5vcnRoIFBhcmsgRHJpdmUsIFN0cmVldGVydmlsbGUsIENoaWNhZ28sIENvb2sgQ291bnR5LCBJbGxpbm9pcywgNjA2MTEsIFVuaXRlZCBTdGF0ZXMgb2YgQW1lcmljYSAtJmd0OyAodG90YWwgcGljayB1cHMgID0gNzM0Mik8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzg2ODk4ZjNjYzA0YjQwY2FhMWUzYWMzNzVmZjBmZGM1LnNldENvbnRlbnQoaHRtbF9jYzhiNjVmYmVjMjc0ZDEyOTI2MTdkZTU5NWY5NzkxMyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9hZmMxYjA5ZWY5MTI0MzFkOWM4NjM2MWM1ZGRiYTk5YS5iaW5kUG9wdXAocG9wdXBfODY4OThmM2NjMDRiNDBjYWExZTNhYzM3NWZmMGZkYzUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYjczZDhiOWVhMmY5NDA4OWFhNjUwMjUzOWVmN2FhM2YgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0MS44OTMyMTYzNiwtODcuNjM3ODQ0MjFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiZ3JlZW4iLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICJncmVlbiIsCiAgImZpbGxPcGFjaXR5IjogMC4yLAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNi41NTUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNzA5YjcyOGY3NjVlNDE1NDhkOWFkMmU1NTgwNGQyMjMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZmZlZDU2NGY3NTI1NGM4YmI3ZmI4N2FiNjQyOTA0ZTMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNDc0NzhkZTY4NzA2NDA1N2ExMTFmYjkxMGM0OGNmNjkgPSAkKCc8ZGl2IGlkPSJodG1sXzQ3NDc4ZGU2ODcwNjQwNTdhMTExZmI5MTBjNDhjZjY5IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij4zNTAtMzcyLCBXZXN0IE9udGFyaW8gU3RyZWV0LCBDYWJyaW5pLUdyZWVuLCBDaGljYWdvLCBDb29rIENvdW50eSwgSWxsaW5vaXMsIDYwNjA3LCBVbml0ZWQgU3RhdGVzIG9mIEFtZXJpY2EgLSZndDsgKHRvdGFsIHBpY2sgdXBzICA9IDY1NTUpPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9mZmVkNTY0Zjc1MjU0YzhiYjdmYjg3YWI2NDI5MDRlMy5zZXRDb250ZW50KGh0bWxfNDc0NzhkZTY4NzA2NDA1N2ExMTFmYjkxMGM0OGNmNjkpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYjczZDhiOWVhMmY5NDA4OWFhNjUwMjUzOWVmN2FhM2YuYmluZFBvcHVwKHBvcHVwX2ZmZWQ1NjRmNzUyNTRjOGJiN2ZiODdhYjY0MjkwNGUzKTsKCiAgICAgICAgICAgIAogICAgICAgIAo8L3NjcmlwdD4=" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>




```python
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
```

    D:\Anaconda\lib\site-packages\ipykernel_launcher.py:16: FutureWarning: convert_objects is deprecated.  Use the data-type specific converters pd.to_datetime, pd.to_timedelta and pd.to_numeric.
      app.launch_new_instance()
    D:\Anaconda\lib\site-packages\ipykernel_launcher.py:17: FutureWarning: convert_objects is deprecated.  Use the data-type specific converters pd.to_datetime, pd.to_timedelta and pd.to_numeric.
    




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><iframe src="data:text/html;charset=utf-8;base64,PCFET0NUWVBFIGh0bWw+CjxoZWFkPiAgICAKICAgIDxtZXRhIGh0dHAtZXF1aXY9ImNvbnRlbnQtdHlwZSIgY29udGVudD0idGV4dC9odG1sOyBjaGFyc2V0PVVURi04IiAvPgogICAgPHNjcmlwdD5MX1BSRUZFUl9DQU5WQVMgPSBmYWxzZTsgTF9OT19UT1VDSCA9IGZhbHNlOyBMX0RJU0FCTEVfM0QgPSBmYWxzZTs8L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS4yLjAvZGlzdC9sZWFmbGV0LmpzIj48L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2FqYXguZ29vZ2xlYXBpcy5jb20vYWpheC9saWJzL2pxdWVyeS8xLjExLjEvanF1ZXJ5Lm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvanMvYm9vdHN0cmFwLm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9jZG5qcy5jbG91ZGZsYXJlLmNvbS9hamF4L2xpYnMvTGVhZmxldC5hd2Vzb21lLW1hcmtlcnMvMi4wLjIvbGVhZmxldC5hd2Vzb21lLW1hcmtlcnMuanMiPjwvc2NyaXB0PgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS4yLjAvZGlzdC9sZWFmbGV0LmNzcyIgLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvY3NzL2Jvb3RzdHJhcC5taW4uY3NzIiAvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL21heGNkbi5ib290c3RyYXBjZG4uY29tL2Jvb3RzdHJhcC8zLjIuMC9jc3MvYm9vdHN0cmFwLXRoZW1lLm1pbi5jc3MiIC8+CiAgICA8bGluayByZWw9InN0eWxlc2hlZXQiIGhyZWY9Imh0dHBzOi8vbWF4Y2RuLmJvb3RzdHJhcGNkbi5jb20vZm9udC1hd2Vzb21lLzQuNi4zL2Nzcy9mb250LWF3ZXNvbWUubWluLmNzcyIgLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9jZG5qcy5jbG91ZGZsYXJlLmNvbS9hamF4L2xpYnMvTGVhZmxldC5hd2Vzb21lLW1hcmtlcnMvMi4wLjIvbGVhZmxldC5hd2Vzb21lLW1hcmtlcnMuY3NzIiAvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL3Jhd2dpdC5jb20vcHl0aG9uLXZpc3VhbGl6YXRpb24vZm9saXVtL21hc3Rlci9mb2xpdW0vdGVtcGxhdGVzL2xlYWZsZXQuYXdlc29tZS5yb3RhdGUuY3NzIiAvPgogICAgPHN0eWxlPmh0bWwsIGJvZHkge3dpZHRoOiAxMDAlO2hlaWdodDogMTAwJTttYXJnaW46IDA7cGFkZGluZzogMDt9PC9zdHlsZT4KICAgIDxzdHlsZT4jbWFwIHtwb3NpdGlvbjphYnNvbHV0ZTt0b3A6MDtib3R0b206MDtyaWdodDowO2xlZnQ6MDt9PC9zdHlsZT4KICAgIAogICAgICAgICAgICA8c3R5bGU+ICNtYXBfNTQ4NDM5YWM0OTc4NDZmMjlhNzUxZGNjOGEyOGNmZDggewogICAgICAgICAgICAgICAgcG9zaXRpb24gOiByZWxhdGl2ZTsKICAgICAgICAgICAgICAgIHdpZHRoIDogMTAwLjAlOwogICAgICAgICAgICAgICAgaGVpZ2h0OiAxMDAuMCU7CiAgICAgICAgICAgICAgICBsZWZ0OiAwLjAlOwogICAgICAgICAgICAgICAgdG9wOiAwLjAlOwogICAgICAgICAgICAgICAgfQogICAgICAgICAgICA8L3N0eWxlPgogICAgICAgIAo8L2hlYWQ+Cjxib2R5PiAgICAKICAgIAogICAgICAgICAgICA8ZGl2IGNsYXNzPSJmb2xpdW0tbWFwIiBpZD0ibWFwXzU0ODQzOWFjNDk3ODQ2ZjI5YTc1MWRjYzhhMjhjZmQ4IiA+PC9kaXY+CiAgICAgICAgCjwvYm9keT4KPHNjcmlwdD4gICAgCiAgICAKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGJvdW5kcyA9IG51bGw7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgdmFyIG1hcF81NDg0MzlhYzQ5Nzg0NmYyOWE3NTFkY2M4YTI4Y2ZkOCA9IEwubWFwKAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgJ21hcF81NDg0MzlhYzQ5Nzg0NmYyOWE3NTFkY2M4YTI4Y2ZkOCcsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB7Y2VudGVyOiBbNDEuOTAzNjgyNDU4MTk5OTk1LC04Ny42NjAzNTI1NzAxXSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHpvb206IDEzLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgbWF4Qm91bmRzOiBib3VuZHMsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBsYXllcnM6IFtdLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgd29ybGRDb3B5SnVtcDogZmFsc2UsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBjcnM6IEwuQ1JTLkVQU0czODU3CiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIH0pOwogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgdGlsZV9sYXllcl8zOGYyYjM0ZWRjNWM0YjVjOWFmN2JmMTFlNGUyNmNjMCA9IEwudGlsZUxheWVyKAogICAgICAgICAgICAgICAgJ2h0dHBzOi8ve3N9LnRpbGUub3BlbnN0cmVldG1hcC5vcmcve3p9L3t4fS97eX0ucG5nJywKICAgICAgICAgICAgICAgIHsKICAiYXR0cmlidXRpb24iOiBudWxsLAogICJkZXRlY3RSZXRpbmEiOiBmYWxzZSwKICAibWF4Wm9vbSI6IDE4LAogICJtaW5ab29tIjogMSwKICAibm9XcmFwIjogZmFsc2UsCiAgInN1YmRvbWFpbnMiOiAiYWJjIgp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF81NDg0MzlhYzQ5Nzg0NmYyOWE3NTFkY2M4YTI4Y2ZkOCk7CiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHRpbGVfbGF5ZXJfMDNiYjUzYjM4ZGJkNGI5NWFhNWYzMTNiNGY2ZTM5NTYgPSBMLnRpbGVMYXllcigKICAgICAgICAgICAgICAgICdodHRwczovL2NhcnRvZGItYmFzZW1hcHMte3N9Lmdsb2JhbC5zc2wuZmFzdGx5Lm5ldC9saWdodF9hbGwve3p9L3t4fS97eX0ucG5nJywKICAgICAgICAgICAgICAgIHsKICAiYXR0cmlidXRpb24iOiBudWxsLAogICJkZXRlY3RSZXRpbmEiOiBmYWxzZSwKICAibWF4Wm9vbSI6IDE4LAogICJtaW5ab29tIjogMSwKICAibm9XcmFwIjogZmFsc2UsCiAgInN1YmRvbWFpbnMiOiAiYWJjIgp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF81NDg0MzlhYzQ5Nzg0NmYyOWE3NTFkY2M4YTI4Y2ZkOCk7CiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNDBjMDY0ZjIyMmRlNGU4Y2I3NTI1MWViYmM4Zjg0MzYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0MS44ODA5OTQ0NzEsLTg3LjYzMjc0NjQ4OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJyZWQiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICJyZWQiLAogICJmaWxsT3BhY2l0eSI6IDAuMiwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDIzLjEsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNTQ4NDM5YWM0OTc4NDZmMjlhNzUxZGNjOGEyOGNmZDgpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNWJmN2Y1YTIxOTZjNGQxOWE4NjEwY2Y2ODA4OGE2ZmQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMDUzYTg4MmJlMzY4NDg2M2E4NzUzN2M3NTNiMjdlZGYgPSAkKCc8ZGl2IGlkPSJodG1sXzA1M2E4ODJiZTM2ODQ4NjNhODc1MzdjNzUzYjI3ZWRmIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Ob3J0aGVybiBUcnVzdCBCYW5rLCA1MCwgU291dGggTGEgU2FsbGUgU3RyZWV0LCBMb29wLCBDaGljYWdvLCBDb29rIENvdW50eSwgSWxsaW5vaXMsIDYwNjAzLCBVbml0ZWQgU3RhdGVzIG9mIEFtZXJpY2EgLSZndDsgKHRvdGFsIGRyb3Agb2ZmcyAgPSAyMzEwMCk8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzViZjdmNWEyMTk2YzRkMTlhODYxMGNmNjgwODhhNmZkLnNldENvbnRlbnQoaHRtbF8wNTNhODgyYmUzNjg0ODYzYTg3NTM3Yzc1M2IyN2VkZik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl80MGMwNjRmMjIyZGU0ZThjYjc1MjUxZWJiYzhmODQzNi5iaW5kUG9wdXAocG9wdXBfNWJmN2Y1YTIxOTZjNGQxOWE4NjEwY2Y2ODA4OGE2ZmQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfY2QxOGM1MGUyMjE2NDlhNThiOTM5NTc3Y2NkYmJkNDcgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0MS44ODQ5ODcxOTIsLTg3LjYyMDk5MjkxM10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJyZWQiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICJyZWQiLAogICJmaWxsT3BhY2l0eSI6IDAuMiwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDEzLjgyLAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzU0ODQzOWFjNDk3ODQ2ZjI5YTc1MWRjYzhhMjhjZmQ4KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzFiMGNmMmExMjE4NzQxNGQ4ZjBmMzdkM2QwMjZjYjFlID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2RjMTQxMmY0NWM3ODRkYThiNzM0OTdkOGU3MjQ2ODNhID0gJCgnPGRpdiBpZD0iaHRtbF9kYzE0MTJmNDVjNzg0ZGE4YjczNDk3ZDhlNzI0NjgzYSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QW9uIENlbnRlciwgMjAwLCBFYXN0IFJhbmRvbHBoIFN0cmVldCwgTG9vcCwgQ2hpY2FnbywgQ29vayBDb3VudHksIElsbGlub2lzLCA2MDYxMSwgVW5pdGVkIFN0YXRlcyBvZiBBbWVyaWNhIC0mZ3Q7ICh0b3RhbCBkcm9wIG9mZnMgID0gMTM4MjApPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8xYjBjZjJhMTIxODc0MTRkOGYwZjM3ZDNkMDI2Y2IxZS5zZXRDb250ZW50KGh0bWxfZGMxNDEyZjQ1Yzc4NGRhOGI3MzQ5N2Q4ZTcyNDY4M2EpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfY2QxOGM1MGUyMjE2NDlhNThiOTM5NTc3Y2NkYmJkNDcuYmluZFBvcHVwKHBvcHVwXzFiMGNmMmExMjE4NzQxNGQ4ZjBmMzdkM2QwMjZjYjFlKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2RhYzdlNDQzOGMwZTQ5ODI4MzZkYTRmMjc0MjIwYWQ4ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDEuODkyNTA3NzgxLC04Ny42MjYyMTQ5MDZdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAicmVkIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAicmVkIiwKICAiZmlsbE9wYWNpdHkiOiAwLjIsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA5LjQ1NSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF81NDg0MzlhYzQ5Nzg0NmYyOWE3NTFkY2M4YTI4Y2ZkOCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9kMjI3OGIzMDcyYzc0ZTkyODgwYTA4MGJhMjM4MjczYiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8wNGIxYTQyMTQxOWM0NTFlOGY3OWQyZGNlZjNiMDM0MyA9ICQoJzxkaXYgaWQ9Imh0bWxfMDRiMWE0MjE0MTljNDUxZThmNzlkMmRjZWYzYjAzNDMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlJ1c2gtT2hpby1XYWJhc2ggU2VsZiBQYXJrLCA1MCwgRWFzdCBPaGlvIFN0cmVldCwgTWFnbmlmaWNlbnQgTWlsZSwgQ2hpY2FnbywgQ29vayBDb3VudHksIElsbGlub2lzLCA2MDYxMSwgVW5pdGVkIFN0YXRlcyBvZiBBbWVyaWNhIC0mZ3Q7ICh0b3RhbCBkcm9wIG9mZnMgID0gOTQ1NSk8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2QyMjc4YjMwNzJjNzRlOTI4ODBhMDgwYmEyMzgyNzNiLnNldENvbnRlbnQoaHRtbF8wNGIxYTQyMTQxOWM0NTFlOGY3OWQyZGNlZjNiMDM0Myk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9kYWM3ZTQ0MzhjMGU0OTgyODM2ZGE0ZjI3NDIyMGFkOC5iaW5kUG9wdXAocG9wdXBfZDIyNzhiMzA3MmM3NGU5Mjg4MGEwODBiYTIzODI3M2IpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMGE4ZWQ0ZDk3Y2VmNDQ3YWE3NDcyNjE2ZGE1MjRiNWQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0MS44OTIwNDIxMzYsLTg3LjYzMTg2Mzk1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInJlZCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogInJlZCIsCiAgImZpbGxPcGFjaXR5IjogMC4yLAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogOS4yNjYsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNTQ4NDM5YWM0OTc4NDZmMjlhNzUxZGNjOGEyOGNmZDgpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMzM0MDM2MjgyMzQ3NDUxOGIxZjQ2MDgxYmQ5ZmI1OGIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNDVlOGE0MTM4MjBhNDJmZDhmODE0Y2Y5ZGNiYzM4NmMgPSAkKCc8ZGl2IGlkPSJodG1sXzQ1ZThhNDEzODIwYTQyZmQ4ZjgxNGNmOWRjYmMzODZjIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5CZXN0IFdlc3Rlcm4gUGx1cyBSaXZlciBOb3J0aCwgMTI1LCBXZXN0IE9oaW8gU3RyZWV0LCBNYWduaWZpY2VudCBNaWxlLCBDaGljYWdvLCBDb29rIENvdW50eSwgSWxsaW5vaXMsIDYwNjExLCBVbml0ZWQgU3RhdGVzIG9mIEFtZXJpY2EgLSZndDsgKHRvdGFsIGRyb3Agb2ZmcyAgPSA5MjY2KTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMzM0MDM2MjgyMzQ3NDUxOGIxZjQ2MDgxYmQ5ZmI1OGIuc2V0Q29udGVudChodG1sXzQ1ZThhNDEzODIwYTQyZmQ4ZjgxNGNmOWRjYmMzODZjKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzBhOGVkNGQ5N2NlZjQ0N2FhNzQ3MjYxNmRhNTI0YjVkLmJpbmRQb3B1cChwb3B1cF8zMzQwMzYyODIzNDc0NTE4YjFmNDYwODFiZDlmYjU4Yik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8wMjMwM2M4NzNiZjc0ZjMyYThmMWZlNGU4M2FhN2M5NSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQxLjk0NDIyNjYwMSwtODcuNjU1OTk4MTgyXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInJlZCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogInJlZCIsCiAgImZpbGxPcGFjaXR5IjogMC4yLAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogOC41MTgsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNTQ4NDM5YWM0OTc4NDZmMjlhNzUxZGNjOGEyOGNmZDgpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYTMzYmM3YTIyMDdhNDQ4YThjNDdkM2RkYTYyMzg0ZjEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNzZjOGE1YWRjYmQ0NGUwZDgwNTVmMjY1NzE4ZTdmYjcgPSAkKCc8ZGl2IGlkPSJodG1sXzc2YzhhNWFkY2JkNDRlMGQ4MDU1ZjI2NTcxOGU3ZmI3IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij4xMDQxLTEwNDMsIFdlc3QgTmV3cG9ydCBBdmVudWUsIFdyaWdsZXl2aWxsZSwgVXB0b3duLCBDaGljYWdvLCBDb29rIENvdW50eSwgSWxsaW5vaXMsIDYwNjU3LCBVbml0ZWQgU3RhdGVzIG9mIEFtZXJpY2EgLSZndDsgKHRvdGFsIGRyb3Agb2ZmcyAgPSA4NTE4KTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYTMzYmM3YTIyMDdhNDQ4YThjNDdkM2RkYTYyMzg0ZjEuc2V0Q29udGVudChodG1sXzc2YzhhNWFkY2JkNDRlMGQ4MDU1ZjI2NTcxOGU3ZmI3KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzAyMzAzYzg3M2JmNzRmMzJhOGYxZmU0ZTgzYWE3Yzk1LmJpbmRQb3B1cChwb3B1cF9hMzNiYzdhMjIwN2E0NDhhOGM0N2QzZGRhNjIzODRmMSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl85N2U5MjM3OTZjZjY0ODdhOGQ3NmNmODk0ZDFiMTQ1YSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQxLjg3OTI1NTA4NCwtODcuNjQyNjQ4OTk4XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInJlZCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogInJlZCIsCiAgImZpbGxPcGFjaXR5IjogMC4yLAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogOC4xNiwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF81NDg0MzlhYzQ5Nzg0NmYyOWE3NTFkY2M4YTI4Y2ZkOCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9jNmE4NTU0M2ZkMzY0YmM3YWFhZWViMWRlZWM0ZWFmOSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9mZGFmMGMxZTM4NWQ0ZjM4OWMzNzZlNDI1MzhhNzViOSA9ICQoJzxkaXYgaWQ9Imh0bWxfZmRhZjBjMWUzODVkNGYzODljMzc2ZTQyNTM4YTc1YjkiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkFsJiMzOTtzICMxIEl0YWxpYW4gQmVlZiwgNjAxLCBXZXN0IEFkYW1zIFN0cmVldCwgR3JlZWt0b3duLCBDaGljYWdvLCBDb29rIENvdW50eSwgSWxsaW5vaXMsIDYwNjYxLCBVbml0ZWQgU3RhdGVzIG9mIEFtZXJpY2EgLSZndDsgKHRvdGFsIGRyb3Agb2ZmcyAgPSA4MTYwKTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYzZhODU1NDNmZDM2NGJjN2FhYWVlYjFkZWVjNGVhZjkuc2V0Q29udGVudChodG1sX2ZkYWYwYzFlMzg1ZDRmMzg5YzM3NmU0MjUzOGE3NWI5KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzk3ZTkyMzc5NmNmNjQ4N2E4ZDc2Y2Y4OTRkMWIxNDVhLmJpbmRQb3B1cChwb3B1cF9jNmE4NTU0M2ZkMzY0YmM3YWFhZWViMWRlZWM0ZWFmOSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8zZmRmY2E0OWY4Nzk0ZjIyOWVjMTlmMjI0YWM2ZDUxYiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQxLjk3OTA3MDgyLC04Ny45MDMwMzk2NjEwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJyZWQiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICJyZWQiLAogICJmaWxsT3BhY2l0eSI6IDAuMiwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDcuNTgzLAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzU0ODQzOWFjNDk3ODQ2ZjI5YTc1MWRjYzhhMjhjZmQ4KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzE3YTIwZjE5Y2E5ODRhYTg4NWQxNGY0ZGNkNjZiMjI4ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzE2YWE3MGMxZWE5YjQ5YzA4ZWNmYzFjMmVjNjg5MzMyID0gJCgnPGRpdiBpZD0iaHRtbF8xNmFhNzBjMWVhOWI0OWMwOGVjZmMxYzJlYzY4OTMzMiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QnVzIERpc3BhdGNoLCBPJiMzOTtIYXJlIEFycml2YWxzLCBPJiMzOTtIYXJlLCBDaGljYWdvLCBDb29rIENvdW50eSwgSWxsaW5vaXMsIDYwMDE4LCBVbml0ZWQgU3RhdGVzIG9mIEFtZXJpY2EgLSZndDsgKHRvdGFsIGRyb3Agb2ZmcyAgPSA3NTgzKTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMTdhMjBmMTljYTk4NGFhODg1ZDE0ZjRkY2Q2NmIyMjguc2V0Q29udGVudChodG1sXzE2YWE3MGMxZWE5YjQ5YzA4ZWNmYzFjMmVjNjg5MzMyKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzNmZGZjYTQ5Zjg3OTRmMjI5ZWMxOWYyMjRhYzZkNTFiLmJpbmRQb3B1cChwb3B1cF8xN2EyMGYxOWNhOTg0YWE4ODVkMTRmNGRjZDY2YjIyOCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl83ODI5ZDE4MGZmZWQ0ZDQ1OWU4YzMwN2ZhYmMyMTI3YyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQxLjg5MDkyMjAyNiwtODcuNjE4ODY4MzU1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInJlZCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogInJlZCIsCiAgImZpbGxPcGFjaXR5IjogMC4yLAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNy40MjIsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNTQ4NDM5YWM0OTc4NDZmMjlhNzUxZGNjOGEyOGNmZDgpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZjE0ZTQ5OTYxMjIyNGNlNDlhZDQyMTE3ZGMwNTU0NjUgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNTQ1MWRkNjdlZGE3NGZiNzkwMzJkMzBjZjM0NDA4MzQgPSAkKCc8ZGl2IGlkPSJodG1sXzU0NTFkZDY3ZWRhNzRmYjc5MDMyZDMwY2YzNDQwODM0IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij40NjUsIE5vcnRoIFBhcmsgRHJpdmUsIFN0cmVldGVydmlsbGUsIENoaWNhZ28sIENvb2sgQ291bnR5LCBJbGxpbm9pcywgNjA2MTEsIFVuaXRlZCBTdGF0ZXMgb2YgQW1lcmljYSAtJmd0OyAodG90YWwgZHJvcCBvZmZzICA9IDc0MjIpPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9mMTRlNDk5NjEyMjI0Y2U0OWFkNDIxMTdkYzA1NTQ2NS5zZXRDb250ZW50KGh0bWxfNTQ1MWRkNjdlZGE3NGZiNzkwMzJkMzBjZjM0NDA4MzQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNzgyOWQxODBmZmVkNGQ0NTllOGMzMDdmYWJjMjEyN2MuYmluZFBvcHVwKHBvcHVwX2YxNGU0OTk2MTIyMjRjZTQ5YWQ0MjExN2RjMDU1NDY1KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2EwNzM0OWI2MWY5MjRmYmE4YzFkMGNkZDg4MmU3MmM3ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDEuODk5NjAyMTExLC04Ny42MzMzMDgwMzddLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAicmVkIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAicmVkIiwKICAiZmlsbE9wYWNpdHkiOiAwLjIsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA3LjEzMiwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF81NDg0MzlhYzQ5Nzg0NmYyOWE3NTFkY2M4YTI4Y2ZkOCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8xYTFkZWM3ZGNjMTE0NjQ2OGUxZTY2MGQ4ZDIwNzBjYyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9lMWJiOThkMGE0ZTY0MjAwOWJjYWJhOGE1YmI1Zjc0YSA9ICQoJzxkaXYgaWQ9Imh0bWxfZTFiYjk4ZDBhNGU2NDIwMDliY2FiYThhNWJiNWY3NGEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkRyeWVyIEhhbGwgLSBNb29keSBCaWJsZSBJbnN0aXR1dGUsIDkzMCwgTm9ydGggTGEgU2FsbGUgQm91bGV2YXJkLCBOZWFyIE5vcnRoIFNpZGUsIENoaWNhZ28sIENvb2sgQ291bnR5LCBJbGxpbm9pcywgNjA2MTEsIFVuaXRlZCBTdGF0ZXMgb2YgQW1lcmljYSAtJmd0OyAodG90YWwgZHJvcCBvZmZzICA9IDcxMzIpPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8xYTFkZWM3ZGNjMTE0NjQ2OGUxZTY2MGQ4ZDIwNzBjYy5zZXRDb250ZW50KGh0bWxfZTFiYjk4ZDBhNGU2NDIwMDliY2FiYThhNWJiNWY3NGEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYTA3MzQ5YjYxZjkyNGZiYThjMWQwY2RkODgyZTcyYzcuYmluZFBvcHVwKHBvcHVwXzFhMWRlYzdkY2MxMTQ2NDY4ZTFlNjYwZDhkMjA3MGNjKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2VkYzgzM2RlZWRjYTRiMzZhOWFmNTZkYTVlODFmMjQ1ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDEuODkzMjE2MzYsLTg3LjYzNzg0NDIxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInJlZCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogInJlZCIsCiAgImZpbGxPcGFjaXR5IjogMC4yLAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNi45MSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF81NDg0MzlhYzQ5Nzg0NmYyOWE3NTFkY2M4YTI4Y2ZkOCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8yYjg5YTJiOTBlNzc0YWNkYjE1MTFmYmNlMDAzZDcxOCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9iMTM5ZjdkZDgxY2I0ZjA2OGViOThlODc0ZjA0NDRjYyA9ICQoJzxkaXYgaWQ9Imh0bWxfYjEzOWY3ZGQ4MWNiNGYwNjhlYjk4ZTg3NGYwNDQ0Y2MiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPjM1MC0zNzIsIFdlc3QgT250YXJpbyBTdHJlZXQsIENhYnJpbmktR3JlZW4sIENoaWNhZ28sIENvb2sgQ291bnR5LCBJbGxpbm9pcywgNjA2MDcsIFVuaXRlZCBTdGF0ZXMgb2YgQW1lcmljYSAtJmd0OyAodG90YWwgZHJvcCBvZmZzICA9IDY5MTApPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8yYjg5YTJiOTBlNzc0YWNkYjE1MTFmYmNlMDAzZDcxOC5zZXRDb250ZW50KGh0bWxfYjEzOWY3ZGQ4MWNiNGYwNjhlYjk4ZTg3NGYwNDQ0Y2MpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZWRjODMzZGVlZGNhNGIzNmE5YWY1NmRhNWU4MWYyNDUuYmluZFBvcHVwKHBvcHVwXzJiODlhMmI5MGU3NzRhY2RiMTUxMWZiY2UwMDNkNzE4KTsKCiAgICAgICAgICAgIAogICAgICAgIAo8L3NjcmlwdD4=" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>


