#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# # LOAD DATA

# In[2]:


circuits = pd.read_csv('dataset/circuits.csv')  #
constructors = pd.read_csv('dataset/constructors.csv') #
constructor_results = pd.read_csv('dataset/constructor_results.csv')
constructor_standings = pd.read_csv('dataset/constructor_standings.csv')
drivers = pd.read_csv('dataset/drivers.csv') #
driver_standings = pd.read_csv('dataset/driver_standings.csv')
lap_times = pd.read_csv('dataset/lap_times.csv') #
pit_stops = pd.read_csv('dataset/pit_stops.csv')  # 
qualifying = pd.read_csv('dataset/qualifying.csv') 
races = pd.read_csv('dataset/races.csv')  #
results = pd.read_csv('dataset/results.csv') #
seasons = pd.read_csv('dataset/seasons.csv')
sprint_results = pd.read_csv('dataset/sprint_results.csv')
status = pd.read_csv('dataset/status.csv')


# # CREATE DF

# In[3]:


lap_times.rename(columns={
    'time': 'lap_time',
    'milliseconds': 'lap_time_ms'
}, inplace=True)
lap_times.drop(columns=['lap_time'], inplace=True)
lap_times


# In[4]:


df = lap_times.copy()


# In[5]:


df


# In[6]:


races = races.drop(columns=['fp1_date', 'fp1_time', 'fp2_date', 'fp2_time', 'fp3_date', 'fp3_time',
       'quali_date', 'quali_time', 'sprint_date', 'sprint_time', 'url'])


# In[7]:


races.rename(columns={
    'name': 'race_name',
    'date': 'race_date',
    'time': 'race_time',
    'year': 'race_year'
}, inplace=True)
races = races[races['race_year'] >= 2010]


# In[8]:


races.info()


# In[9]:


df = df.merge(races, on='raceId', how='inner')


# In[10]:


df.info()


# In[11]:


pit_stops_renamed = pit_stops.rename(columns={'lap': 'pit_lap'})
df = df.merge(
    pit_stops_renamed[['raceId', 'driverId', 'pit_lap']],
    left_on=['raceId', 'driverId', 'lap'],
    right_on=['raceId', 'driverId', 'pit_lap'],
    how='left'
)


# In[12]:


df


# In[13]:


df.info()


# In[14]:


df['pit_stop'] = df['pit_lap'].notna().astype(int)
df.drop(columns=['pit_lap'], inplace=True)


# In[15]:


df['tyre_age'] = 0
df = df.sort_values(['raceId', 'driverId', 'lap'])
for (race, driver), group in df.groupby(['raceId', 'driverId']):
    last_pit = 0
    ages = []

    for lap, pit in zip(group['lap'], group['pit_stop']):
        if pit == 1:
            last_pit = lap
        ages.append(lap - last_pit)

    df.loc[group.index, 'tyre_age'] = ages


# In[16]:


df


# In[17]:


df = df.merge(
    results[['raceId', 'driverId', 'constructorId', 'grid', 'positionOrder']],
    on=['raceId', 'driverId'],
    how='left'
)


# In[18]:


df


# In[19]:


df = df.merge(
    circuits[['circuitId', 'lat', 'lng', 'alt']],
    on='circuitId',
    how='left'
)
df


# In[20]:


df.to_csv("f1.csv")


# # EDA

# In[20]:


df.isnull().sum()


# In[21]:


df.isna().sum()


# In[22]:


df.describe()
df['lap_time_ms'].median()


# In[23]:


df.columns


# In[24]:


df.info()


# In[25]:


numerical_cols = [
    'lap', 'lap_time_ms', 'race_year', 'circuitId', 'pit_stop', 'tyre_age', 
]


# In[26]:


for col in numerical_cols:
    plt.figure(figsize=(8, 6))
    sns.histplot(df[col], kde=True, bins=20)


# In[27]:


for col in numerical_cols:
    plt.figure(figsize=(10, 8))
    sns.boxplot( x = df[col])


# In[28]:


df = df[df['lap_time_ms'] <= 120000]


# In[29]:


plt.figure(figsize=(10, 8))
sns.boxplot( x = df['lap_time_ms'])


# In[30]:


plt.figure(figsize=(20, 16))
sns.heatmap(df.corr(numeric_only=True), annot=True)


# In[31]:


df.shape


# In[32]:


df.drop_duplicates(inplace=True)


# In[33]:


df.shape


# # FEATURE ENG AND EXT

# In[34]:


df['grid'] = pd.to_numeric(df['grid'], errors='coerce')
df = df.dropna(subset=['grid'])


# In[35]:


train_df = df[df['race_year'] < 2020]
test_df  = df[df['race_year'] >= 2020]

driver_avg = train_df.groupby('driverId')['lap_time_ms'].mean()

train_df['driver_skill'] = train_df['driverId'].map(driver_avg)
test_df['driver_skill']  = test_df['driverId'].map(driver_avg)

test_df['driver_skill'] = test_df['driver_skill'].fillna(driver_avg.mean())


# In[36]:


circuit_avg = train_df.groupby('circuitId')['lap_time_ms'].mean()

train_df['circuit_difficulty'] = train_df['circuitId'].map(circuit_avg)
test_df['circuit_difficulty']  = test_df['circuitId'].map(circuit_avg)

test_df['circuit_difficulty'] = test_df['circuit_difficulty'].fillna(circuit_avg.mean())
df


# In[37]:


train_df['prev_lap_time'] = train_df.groupby(['raceId','driverId'])['lap_time_ms'].shift(1)
test_df['prev_lap_time']  = test_df.groupby(['raceId','driverId'])['lap_time_ms'].shift(1)

train_df = train_df.dropna()
test_df  = test_df.dropna()


# In[38]:


df.columns


# In[39]:


drop_cols = [
    'race_name', 'race_date', 'race_time', 'raceId', 'drverId', 'circuitId', 'constructorId'
]


# In[40]:


features = [
    'lap',
    'position',
    'pit_stop',
    'tyre_age',
    'grid',
    'alt',
    'driver_skill',
    'circuit_difficulty',
    'race_year',
    'round',
    'prev_lap_time'
]


# In[ ]:





# In[41]:


df.info()


# In[42]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

X_train = train_df[features]
y_train = train_df['lap_time_ms']

X_test = test_df[features]
y_test = test_df['lap_time_ms']

model = RandomForestRegressor(n_estimators=100, n_jobs=-1)
model.fit(X_train, y_train)


# In[43]:


from sklearn.metrics import r2_score, mean_absolute_error

y_pred = model.predict(X_test)

print("R2:", r2_score(y_test, y_pred))
print("MAE (sec):", mean_absolute_error(y_test, y_pred)/1000)


# In[44]:


y_test


# In[45]:


y_pred


# # KNN

# In[46]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)


# In[47]:


from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error

knn = KNeighborsRegressor(
    n_neighbors=40,  
    n_jobs=-1
)
knn.fit(X_train_scaled, y_train)


# In[48]:


y_pred_knn = knn.predict(X_test_scaled)

r2_knn  = r2_score(y_test, y_pred_knn)
mae_knn = mean_absolute_error(y_test, y_pred_knn)/1000  

print("knn R2:", r2_knn)
print("knn MAE (sec):", mae_knn)


# # LINEAR REGG

# In[49]:


from sklearn.linear_model import LinearRegression


# In[50]:


lr = LinearRegression()
lr.fit(X_train_scaled, y_train)


# In[51]:


y_pred_lr = lr.predict(X_test_scaled)

r2_lr  = r2_score(y_test, y_pred_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)/1000  

print("linear regression R2:", r2_lr)
print("linear regression mae (sec):", mae_lr)


# In[ ]:




