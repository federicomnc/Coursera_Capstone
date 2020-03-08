#!/usr/bin/env python
# coding: utf-8

# In[1]:


import types
import pandas as pd
from botocore.client import Config
import ibm_boto3

def __iter__(self): return 0

# @hidden_cell
# The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
# You might want to remove those credentials before you share the notebook.
client_6816ff6b00594052973bfb2ff7e1d9ef = ibm_boto3.client(service_name='s3',
    ibm_api_key_id='zZPJmXacX0LovJaeQl1LBHKu6x0ClB3gOqToJXmmpG6E',
    ibm_auth_endpoint="https://iam.eu-gb.bluemix.net/oidc/token",
    config=Config(signature_version='oauth'),
    endpoint_url='https://s3.eu-geo.objectstorage.service.networklayer.com')

body = client_6816ff6b00594052973bfb2ff7e1d9ef.get_object(Bucket='courseraapplieddatasciencecapston-donotdelete-pr-zcajfenqq0r45n',Key='Toronto Neighborhoods.xlsx')['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

df_data_0 = pd.read_excel(body)
df_data_0.head()


# In[2]:


df_data_0.columns=['PostalCode', 'Borough', 'Neighborhood']


# In[3]:


df_assigned=df_data_0[df_data_0['Borough']!='Not assigned'].reset_index().drop('index', axis=1)


# In[4]:


df_sorted=df_assigned.sort_values(by =['PostalCode']).reset_index().drop('index', axis=1)


# In[5]:


#column_names = ["a", "b", "c"]

#df = pd.DataFrame(columns = column_names)
df_sorted.head()


# In[6]:


dfcompact=df_sorted.groupby(['PostalCode','Borough'])['Neighborhood'].apply(','.join).reset_index()


# column_names = ['PostalCode', 'Borough', 'Neighborhood']
# 
# dfcompact = pd.DataFrame(columns = column_names)
# 
# for postalcode in set(df_sorted['PostalCode']):
#     Neighborhoods = ''
#     for i in range(df_sorted[df_sorted['PostalCode']==postalcode].shape[0]):
#         Neighborhoods=Neighborhoods + ', ' + df_sorted[df_sorted['PostalCode']==postalcode].iloc[i]['Neighborhood']
#     dfcompact=dfcompact.append({'PostalCode': postalcode, 
#                                 'Borough': df_sorted[df_sorted['PostalCode']==postalcode].iloc[0]['Borough'],
#                                 'Neighborhood': Neighborhoods.strip(',')}, ignore_index=True)

# In[7]:


dfcompact.head()


# In[8]:


body = client_6816ff6b00594052973bfb2ff7e1d9ef.get_object(Bucket='courseraapplieddatasciencecapston-donotdelete-pr-zcajfenqq0r45n',Key='Geospatial_Coordinates.csv')['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

df_geo = pd.read_csv(body)
df_geo.head()


# In[9]:


df_comp_geo=dfcompact.join(df_geo.set_index('Postal Code'), on='PostalCode')


# In[10]:


df_comp_geo.head()


# In[11]:


neighborhoods=df_comp_geo[df_comp_geo['Borough'].apply(lambda x: x.find('Toronto')!=-1)].reset_index().drop('index', axis=1)


# In[12]:


neighborhoods.head()


# In[13]:


import numpy as np # library to handle data in a vectorized manner

import json # library to handle JSON files

get_ipython().system("conda install -c conda-forge geopy --yes # uncomment this line if you haven't completed the Foursquare API lab")
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values

import requests # library to handle requests
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

# import k-means from clustering stage
from sklearn.cluster import KMeans

get_ipython().system("conda install -c conda-forge folium=0.5.0 --yes # uncomment this line if you haven't completed the Foursquare API lab")
import folium # map rendering library

print('Libraries imported.')


# In[14]:


address = 'Toronto'

geolocator = Nominatim(user_agent="to_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Toronto are {}, {}.'.format(latitude, longitude))


# In[15]:


# create map of Toronto using latitude and longitude values
map_Toronto = folium.Map(location=[latitude, longitude], zoom_start=10)

# add markers to map
for lat, lng, borough, neighborhood in zip(neighborhoods['Latitude'], neighborhoods['Longitude'], neighborhoods['Borough'], neighborhoods['Neighborhood']):
    label = '{}, {}'.format(neighborhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_Toronto)  
    
map_Toronto


# In[16]:


CLIENT_ID = 'LULHM4VRGYYNRP0UCEDDU3M2GPA55IQ4UH5142KJIHOSKR4W' # your Foursquare ID
CLIENT_SECRET = '2ERUN2D0NN4T1PJ2VAVONFFEJWHG21IPUBM23N4XVHI0MP0I' # your Foursquare Secret
VERSION = '20180605' # Foursquare API version

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)


# In[17]:


radius = 500
LIMIT = 100


# In[18]:


def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


# In[19]:


Toronto_venues = getNearbyVenues(names=neighborhoods['Neighborhood'],
                                   latitudes=neighborhoods['Latitude'],
                                   longitudes=neighborhoods['Longitude']
                                  )


# In[20]:


print(Toronto_venues.shape)
Toronto_venues.head()


# In[21]:


Toronto_venues.groupby('Neighborhood').count()


# In[22]:


print('There are {} uniques categories.'.format(len(Toronto_venues['Venue Category'].unique())))


# In[23]:


# one hot encoding
Toronto_onehot = pd.get_dummies(Toronto_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
Toronto_onehot['Neighborhood'] = Toronto_venues['Neighborhood'] 

# move neighborhood column to the first column
fixed_columns = [Toronto_onehot.columns[-1]] + list(Toronto_onehot.columns[:-1])
Toronto_onehot = Toronto_onehot[fixed_columns]

Toronto_onehot.head()


# In[24]:


Toronto_onehot.shape


# In[25]:


Toronto_grouped = Toronto_onehot.groupby('Neighborhood').mean().reset_index()
Toronto_grouped


# In[26]:


Toronto_grouped.shape


# In[27]:


num_top_venues = 5

for hood in Toronto_grouped['Neighborhood']:
    print("----"+hood+"----")
    temp = Toronto_grouped[Toronto_grouped['Neighborhood'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')


# In[28]:


def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# In[29]:


num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = Toronto_grouped['Neighborhood']

for ind in np.arange(Toronto_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(Toronto_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()


# In[30]:


# set number of clusters
kclusters = 5

Toronto_grouped_clustering = Toronto_grouped.drop('Neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(Toronto_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10] 


# In[31]:


# add clustering labels
neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

Toronto_merged = neighborhoods

# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
Toronto_merged = Toronto_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighborhood')

Toronto_merged.head() # check the last columns!


# In[32]:


# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(Toronto_merged['Latitude'], Toronto_merged['Longitude'], Toronto_merged['Neighborhood'], Toronto_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# In[33]:


Toronto_merged.loc[Toronto_merged['Cluster Labels'] == 0, Toronto_merged.columns[[1] + list(range(5, Toronto_merged.shape[1]))]]


# In[34]:


Toronto_merged.loc[Toronto_merged['Cluster Labels'] == 1, Toronto_merged.columns[[1] + list(range(5, Toronto_merged.shape[1]))]]


# In[35]:


Toronto_merged.loc[Toronto_merged['Cluster Labels'] == 2, Toronto_merged.columns[[1] + list(range(5, Toronto_merged.shape[1]))]]


# In[36]:


Toronto_merged.loc[Toronto_merged['Cluster Labels'] == 3, Toronto_merged.columns[[1] + list(range(5, Toronto_merged.shape[1]))]]


# In[37]:


Toronto_merged.loc[Toronto_merged['Cluster Labels'] == 4, Toronto_merged.columns[[1] + list(range(5, Toronto_merged.shape[1]))]]


# In[ ]:




