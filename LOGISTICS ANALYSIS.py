#LOGISTICS
from pickle import FALSE
from re import A

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns
from email._header_value_parser import RouteComponentMarker
from IPython.display import display

df = pd.read_csv(r"C:\Users\HP\Desktop\PROJECT  WORKS\PYTHON\LOGISTICS ANALYIS WITH PYTHON\logistics_dirty_dataset.csv")
print(df)
#Tring to understand my data and what to fix and clean in the data set
print(df.head())
print(df.shape)
print(df.info())
print(df.describe(include='all'))


#checking all categorical columns
print(df['origin_city'].unique())
#apply title case to the origin city column
df['origin_city'] = df['origin_city'].str.title()
print(df['origin_city'].unique())
#apply title case to all column
print(df['origin_city'].unique())
print(df['destination_city'].unique())
print(df['vehicle_type'].unique())
print(df['weather_condition'].unique())
print(df['day_of_week'].unique())
print(df['month'].unique())
print(df['traffic_level'].unique())


for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].str.title()

#checking my columns
for col in df.columns:
    print(col)
    
#checking for duplicates 
print(df.duplicated().sum())
df = df.drop_duplicates() 
print(df.shape)
#checking if there are missing values
print(df.isnull().sum())
#CHECKING THE DISTRIBUTION OF KM
df[['distance_km']].hist()
plt.show()

#fill up with median
df['distance_km'].fillna(df['distance_km'].median(),inplace=True)
print(df.isnull().sum())

#checking the distribution of delivery cost
df['delivery_cost'].hist()
plt.show()
#fill up with median
df['delivery_cost'].fillna(df['delivery_cost'].median(),inplace=True)
print(df.isnull().sum())
#checking the distribution of delivery cost
df['customer_rating'].hist()
plt.show()
#fill up with median
df['customer_rating'].fillna(df['customer_rating'].median(),inplace=True)
print(df.isnull().sum())

#adding missing indicators
df['distance_missing'] = df['distance_km'].isnull().astype(int)
df['cost_missing'] = df['delivery_cost'].isnull().astype(int)
df['rating_missing'] = df['customer_rating'].isnull().astype(int)

#fix data types
df.info()
#convert order date and delivery date to datetime data type
df['order_date'] = pd.to_datetime(df['order_date'])
df['delivery_date'] = pd.to_datetime(df['delivery_date'])

df.info()
#convert the categorical column to the right data type
df['origin_city'] = df['origin_city'].astype('category')
df['destination_city'] = df['destination_city'].astype('category')
df['vehicle_type'] = df['vehicle_type'].astype('category')
df['weather_condition'] = df['weather_condition'].astype('category')
df['traffic_level'] = df['traffic_level'].astype('category')
df['day_of_week'] = df['day_of_week'].astype('category')
df.info()
#fixing invaild values
print(df[df['delivery_time_hours'] < 0])
print((df['delivery_time_hours'] < 0).sum())

df[['delivery_time_hours']].hist()
plt.show()

#fix
print(df[df['delivery_time_hours'] >= 0])

df = df.drop(df[df['delivery_time_hours'] < 0].index)
print((df['delivery_time_hours'] < 0).sum())
#delivery before order date 
print(df[df['delivery_date'] < df['order_date']])
print((df['delivery_date'] < df['order_date']).sum())

#fix

df = df.drop(df[df['delivery_date'] < df['order_date']].index)
print((df['delivery_date'] < df['order_date']).sum())
print(df.shape)

#standardize categorical values
df['origin_city'] = df['origin_city'].str.title()
df['destination_city'] = df['destination_city'].str.title()
df['vehicle_type'] = df['vehicle_type'].str.capitalize()

#handle outliers
import seaborn as sns
sns.boxplot(df['distance_km'])
plt.show()
print(df.describe())

#iqr method
Q1 = df['distance_km'].quantile(0.25)
Q3 = df['distance_km'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
df = df[(df['distance_km'] >= lower) & (df['distance_km'] <= upper)]

#deliver_cost
Q1 = df['delivery_cost'].quantile(0.25)
Q3 = df['delivery_cost'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
df = df[(df['delivery_cost'] >= lower) & (df['delivery_cost'] <= upper)]

#delivery_time_hours
Q1 = df['delivery_time_hours'].quantile(0.25)
Q3 = df['delivery_time_hours'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
df = df[(df['delivery_time_hours'] >= lower) & (df['delivery_time_hours'] <= upper)]
print(df.describe())
#step 8 check consistency
df[df['origin_city'] == df['destination_city']]
print(df.info())
print(df.isnull().sum())
print(df.describe())
#professional edge to make my work stand out
df['data_quality_flag'] = (
    (df['delivery_time_hours'] < 0) |
    (df['delivery_date'] < df['order_date'])
    
).astype(int)

#feature engineering
#step 1 time based features
df['order_day'] = df['order_date'].dt.day
df['order_month'] = df['order_date'].dt.month
df['order_year'] = df['order_date'].dt.year
df['order_day_of_week'] = df['order_date'].dt.day_name()
df['is_weekend'] = df['order_day_of_week'].isin(['Saturday', 'Sunday']).astype(int)
#step 2 delivery performance metrics
df['actual_delivery_time'] = (
    df['delivery_date'] - df['order_date']
).dt.total_seconds() / 3600  # convert seconds to hours

#delivery speed (OTD)
df['delivery_speed_km_per_hr'] = df['distance_km'] / df['actual_delivery_time']
#step 3 cost efficiency features
#cost per km
df['cost_per_km'] = df['delivery_cost'] / df['distance_km']
#cost per hour
df['cost_per_hour'] = df['delivery_cost'] / df['actual_delivery_time']
#handle divide by zero safely
df.replace([np.inf, -np.inf], np.nan, inplace=True)

#STEP 4 DELAY ANALYSIS FEATURES
#define delay (business rule: if actual delivery time > promised delivery time, then delay)
df['is_delayed'] = (df['actual_delivery_time'] > 24).astype(int)  # assuming promised delivery time is 24 hours
#delay magnitude
df['delay_hours'] = df['actual_delivery_time'] - 24
df['delay_hours'] = df['delay_hours'].apply(lambda x: x if x > 0 else 0)  # set negative delays to 0
#step 5 route intelligence
#route identifier
df['route'] = df['origin_city'].astype(str) + ' to ' + df['destination_city'].astype(str)
print(df.head())
#same city indicator
df['same_city_delivery'] = (df['origin_city'] == df['destination_city']).astype(int)
print(df.head())

#step 6 vehicle performance features 
#efficiency by vehicle
vehicle_avg_speed = df.groupby('vehicle_type')['delivery_speed_km_per_hr'].transform('mean')
print(df.head())
df['vehicle_efficiency_vs_avg'] = df['delivery_speed_km_per_hr'] / vehicle_avg_speed
print(df.head())
#cost vs vehicle benchmark
vehicle_avg_cost = df.groupby('vehicle_type')['cost_per_km'].transform('mean')
print(df.head())
df['cost_efficiency_vs_vehicle'] = df['cost_per_km'] / vehicle_avg_cost
#step 7 customer experience features
#rating buckets
df['rating_category'] = pd.cut(
    df['customer_rating'],
    bins=[0, 2, 3.5, 5],
    labels=['Poor', 'Average', 'Good']
)
#low satisfaction flag
df['low_rating_flag'] = (df['customer_rating'] < 3).astype(int)
#step 8 distance segmentation
df['distance_category'] = pd.cut(
    df['distance_km'],
    bins=[0, 50, 200, 500, 2000],
    labels=['Short', 'Medium', 'Long', 'Very Long']
)
#step 9 demand intensity (advanced)
#order per day 
daily_orders = df.groupby('order_date').size()
df['daily_order_volume'] = df['order_date'].map(daily_orders)
#peak demand flag (business rule: if daily orders > 75th percentile, then peak demand)
threshold = df['daily_order_volume'].quantile(0.75)
print(df.head())
df['is_peak_day'] = (df['daily_order_volume'] >= threshold).astype(int)
#step 10 final check
print(df.head())
print(df.describe())

#exploratory data analysis EDA
#SECTION 1 Dataset OVERVIEW
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")
print(df.head())
print(df.describe(include='object'))
#section 2 univariate analysis
#categorical columns
#numerical columns
#plot histograms for distance_km, delivery_cost,actual_delivery_time, customer_rating,cost_per_km, cost_per_hour, delivery_speed_km_per_hr
df['delivery_cost'].hist(bins=30)
plt.title('Distribution of Delivery Cost')
plt.xlabel('Delivery Cost')
plt.ylabel('Frequency')
plt.show()

df['distance_km'].hist(bins=30)
plt.title('Distribution of Distance')
plt.xlabel('Distance (km)')
plt.ylabel('Frequency')
plt.show()

df['actual_delivery_time'].hist(bins=30)
plt.title('Distribution of Actual Delivery Time')
plt.xlabel('Actual Delivery Time (hours)')
plt.ylabel('Frequency')
plt.show()


df['customer_rating'].hist(bins=30)
plt.title('Distribution of Customer Rating')
plt.xlabel('Customer Rating')
plt.ylabel('Frequency')
plt.show()

df['cost_per_km'].hist(bins=30)
plt.title('Distribution of Cost per km')
plt.xlabel('Cost per km')
plt.ylabel('Frequency')
plt.show()

df['cost_per_hour'].hist(bins=30)
plt.title('Distribution of Cost per Hour')
plt.xlabel('Cost per Hour')
plt.ylabel('Frequency')
plt.show()

df['delivery_speed_km_per_hr'].hist(bins=30)
plt.title('Distribution of Delivery Speed')
plt.xlabel('Delivery Speed (km/h)')
plt.ylabel('Frequency')
plt.show()


#qestions to answer
#are costs normally distributed or skewed?
#what is the average delivery cost?
#are there extremely expensive deliveries that might be outliers?
#is delivery time normally distributed or skewed?


#categorical variables 
#count plots (bar charts) for : vehicle_type, weather_condition, traffic_level, day_of_week, month, origin_city, destination_city, delayed
df['vehicle_type'].value_counts().plot(kind='bar')
plt.title("vehicle Distribution")
plt.show()

df['weather_condition'].value_counts().plot(kind='bar')
plt.title("Weather Condition Distribution")
plt.ylabel("Number of Deliveries")
plt.show()



traffic_order = ['Low', 'Medium', 'High']

df['traffic_level'] = pd.Categorical(df['traffic_level'], categories=traffic_order, ordered=True)
df['traffic_level'].value_counts().sort_index().plot(kind='bar')
plt.title("Traffic Level Distribution")
plt.ylabel("Number of Deliveries")
plt.show()

day_order = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
df['day_of_week'] = pd.Categorical(df['day_of_week'], categories=day_order, ordered=True)
df['day_of_week'].value_counts().sort_index().plot(kind='bar')
plt.title("Day of Week Distribution")
plt.ylabel("Number of Deliveries")
plt.show()


df['month'].value_counts().sort_index().plot(kind='bar')
plt.title("Month Distribution")
plt.ylabel("Number of Deliveries")
plt.show()



df['origin_city'].value_counts().plot(kind='bar')
plt.title("Origin City Distribution")
plt.ylabel("Number of Deliveries")
plt.show()


df['destination_city'].value_counts().plot(kind='bar')
plt.title("Destination City Distribution")
plt.ylabel("Number of Deliveries")
plt.show()


df['delayed'].value_counts().plot(kind='bar')
plt.title("Delayed Deliveries Distribution")
plt.ylabel("Number of Deliveries")
plt.show()


#section 3 bivariate analysis
#delivery cost vs distance_km
#scatter plot
# qestions :does not increase with distance , any unusal oberservation
#average cost per vehicle type
df.groupby('vehicle_type')['delivery_cost'].mean().plot(kind='bar')
plt.title("Average Delivery Cost by Vehicle Type")
plt.show()
#which vehicle type is more expensive 
sns.scatterplot(x='distance_km', y='delivery_cost', data=df)
plt.title('Delivery Cost vs Distance')
plt.show()


sns.scatterplot(x='distance_km', y='delivery_time_hours', data=df)
plt.title('Delivery vs Delivery Time')
plt.show()
df[['distance_km', 'delivery_time_hours']].corr()
#using correlation to check distance and dilvery cost
df[['distance_km', 'delivery_cost']].corr()
#average delivery time by vehicle type
sns.boxplot(x='vehicle_type', y='delivery_speed_km_per_hr', data=df)
plt.title('speed by vehicle type')
plt.show()

df.groupby('vehicle_type')['actual_delivery_time'].mean().plot(kind='bar')
plt.title("Average Delivery Time by Vehicle Type")
plt.show()
#customer rateing by vehicle type
df.groupby('vehicle_type')['customer_rating'].mean().plot(kind='bar')
plt.title("Customer Rating by Vehicle Type")
plt.show()
#delay rate by traffic
df.groupby('traffic_level')['is_delayed'].mean().plot(kind='bar')
plt.title("Delay Rate by Traffic Level")
plt.show()

#delay rate by weather condition
df.groupby('weather_condition')['is_delayed'].mean().plot(kind='bar')
plt.title("Delay Rate by Weather Condition")
plt.show()

#average cost by city
df.groupby('origin_city')['delivery_cost'].mean().plot(kind='bar')
plt.title("Average Delivery Cost by Origin City")
plt.show()

#section 4: multivariate analysis
df.groupby('vehicle_type')['delivery_speed_km_per_hr'].mean()
df.groupby('vehicle_type')['delivery_speed_km_per_hr'].mean().plot(kind='bar')
plt.show()
#create a correlation matrix

plt.figure(figsize=(14,10))
sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap='coolwarm',
    fmt='.2f',
    linewidths=0.5
)
plt.title('correlation Matrix of logistics variables')
plt.show()
#3. Understand what the numbers mean
#Correlation ranges from:
#-1 to +1
#Positive correlation
#If you get:
#0.85
#that's a strong positive relationship.
#It means:
#As one variable increases, the other tends to increase.
#For your logistics project:
#Distance ↑ → Delivery Cost ↑
#would make business sense.
#Negative correlation
#If you get:
#-0.35
#that's a moderate negative relationship.
#It means:
#As one variable increases, the other tends to decrease.
#For example:
#Delivery Time ↑ → Customer Rating ↓
#would suggest that longer delivery times are associated with lower customer satisfaction.
#Close to zero
#If you get:
#0.03
#there is very little linear relationship between those variables.
corr = df.select_dtypes(include=['number']).corr()
df['day_of_week_num'] = df['day_of_week'].map({
    'Sunday':1,
    'Monday':2, 
    'Tuesday':3, 
    'Wednesday':4,
    'Thursday':5, 
    'Friday':6, 
    'Saturday':7 
})
#investigate four relationships
print(corr.loc['distance_km','delivery_cost'])
#If you get something like:
#0.82
#you could say:
#Distance has a strong positive correlation with delivery cost, indicating that longer deliveries generally incur higher transportation costs.

#distance vs delivery time
print(corr.loc['distance_km','actual_delivery_time'])
#For example:
#0.70
#Interpretation:
#Distance has a strong positive relationship with delivery time, suggesting that longer routes generally require more time to complete.
#But remember: correlation does not prove causation.


#delay vs rating
print(corr.loc['is_delayed', 'customer_rating'])
#If you get:
#-0.30
#you could interpret it as:
#Delayed deliveries are associated with lower customer ratings, indicating that delivery reliability may have an impact on customer satisfaction.
#But there's something important here.
#Your is_delayed is binary:
#0 = Not delayed
#1 = Delayed
#That's perfectly acceptable for correlation analysis.



#cost vs rating
print(corr.loc['delivery_cost', 'customer_rating'])
#Suppose you get:
#-0.05
#That would suggest:
#There is very little linear relationship between delivery cost and customer rating.
#That's actually an interesting business finding.
#It could mean customers care more about delivery reliability and speed than the absolute delivery cost.
#But don't jump to that conclusion from correlation alone. You would investigate it further with grouped analysis.
# Map weekdays to numbers

# Step 1: Create numeric weekday column
df['day_of_week_num'] = df['day_of_week'].map({
    'Sunday': 1,
    'Monday': 2,
    'Tuesday': 3,
    'Wednesday': 4,
    'Thursday': 5,
    'Friday': 6,
    'Saturday': 7
})

# Step 2: Build correlation matrix AFTER adding the new column
corr = df.select_dtypes(include=['number']).corr()

#8 create a small summary table
key_correlations = pd.DataFrame({
    'Relationship': [
        'Distance vs Delivery Cost',
        'Distance vs Delivery Time',
        'Delay vs Customer Rating',
        'Delivery Cost vs Customer Rating'
    ],
    'correlation':[
        corr.loc['distance_km', 'delivery_cost'],
        corr.loc['distance_km', 'actual_delivery_time'],
        corr.loc['is_delayed', 'customer_rating'],
        corr.loc['delivery_cost', 'customer_rating']
    ]
})
key_correlations

key_correlations['correlation'] = key_correlations['correlation'].round(2)
key_correlations

#orders by day of the week
sns.countplot(x='day_of_week', data=df,
              order=['Sunday','Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'])
plt.title('Order_by_Day_Of_week')
plt.xticks(rotation=45)
plt.show()


#daily demand trend
df.groupby('order_date').size().plot()
plt.title('Daily_order_volume')
plt.show()

#use a scatter plot to see the relationship
#DISTANCE VS COST
plt.figure(figsize=(8,5))
plt.scatter(
    df['distance_km'],
    df['delivery_cost'],
    alpha=0.5
)
plt.title('Distance vs Delivery_Cost')
plt.xlabel('Distance_km')
plt.ylabel('Delivery_Cost')
plt.show()
#You should expect to see the points generally moving upward if distance and cost are positively correlated.
#Distance vs Delivery time
plt.figure(figsize=(8,5))
plt.scatter(
    df['distance_km'],
    df['actual_delivery_time'],
    alpha=0.5
)

plt.title('Distance vs Delivery Time')
plt.xlabel('Distance_Km')
plt.ylabel('Delivery_Time_Hours')
plt.show()
#delivery time vs customer rating
plt.figure(figsize=(8,5))
plt.scatter(
    df['actual_delivery_time'],
    df['customer_rating'],
    alpha=0.5
)

plt.title('Delivery_Time vs Customer_Rating')
plt.xlabel('Delivery_Time_Hours')
plt.ylabel('Customer_Rating')
plt.show()
#This gives you more information because actual_delivery_time is continuous rather than just 0/1.
#cost v rating
plt.figure(figsize=(8, 5))

plt.scatter(
    df['delivery_cost'],
    df['customer_rating'],
    alpha=0.5
)

plt.title('Delivery Cost vs Customer Rating')
plt.xlabel('Delivery Cost')
plt.ylabel('Customer Rating')
plt.show()

#"Distance is strongly positively correlated with delivery cost."
df.groupby('vehicle_type').agg({
    'distance_km':'mean',
    'delivery_cost':'mean',
    'cost_per_km':'mean'
})
#moving from statistical observation → operational analysis.

#EDA flow
#Correlation Matrix
 #       ↓
#Identify Strong Relationships
 #       ↓
#Extract Key Correlations
 #       ↓
#Visualize with Scatter Plots
  #      ↓
#nvestigate by Vehicle / Traffic / Weather / Route
  #      ↓
#Convert Findings into Business Insights
##
corr = df.select_dtypes(include='number').corr()
print(corr)
plt.figure(figsize=(14,10))
sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap='coolwarm',
    fmt='.2f',
    linewidths=0.5
)
plt.title('correlation Matrix of logistics variables')
plt.show()

key_correlations = pd.DataFrame({
    'Relationship': [
        'Distance vs Delivery Cost',
        'Distance vs Delivery Time',
        'Delay vs Customer Rating',
        'Delivery Cost vs Customer Rating'
    ],
    'Correlation': [
        corr.loc['distance_km', 'delivery_cost'],
        corr.loc['distance_km', 'actual_delivery_time'],
        corr.loc['is_delayed', 'customer_rating'],
        corr.loc['delivery_cost', 'customer_rating']
    ]
})

key_correlations['Correlation'] = key_correlations['Correlation'].round(2)

key_correlations


#Delay vs Customer Rating
df.boxplot(
    column='customer_rating',
    by='is_delayed'
)

plt.title('Customer Rating by Delivery Delay Status')
plt.suptitle('')
plt.xlabel('Delayed (0 = No, 1 = Yes)')
plt.ylabel('Customer Rating')
plt.show()


#averages
df.groupby('is_delayed')['customer_rating'].mean()

#section 5 route analysis
route_analysis = df.groupby('route').agg({
    'delivery_cost':'mean',
    'actual_delivery_time':'mean',
    'is_delayed':'mean'
})
route_analysis
#Business questions:
#Which routes are the slowest?
slowest_route = df.groupby('route').agg(
    total_deliveries=('route', 'count'),
    avg_delivery_time=('actual_delivery_time', 'mean')
).query("total_deliveries >= 20").sort_values(
    'avg_delivery_time',
    ascending=False
)
print(slowest_route.head(10))


#This gives you the 10 slowest routes.
slowest_route = df.groupby('route').agg(
    total_deliveries=('route', 'count'),
    avg_delivery_time=('delivery_time_hours', 'mean')
).query("total_deliveries >= 20").sort_values(
    'avg_delivery_time',
    ascending=False
)
print(slowest_route.head(10))


#Show me the slowest routes among routes with at least 20 deliveries."
#Which routes are most expensive?
most_expensive_route = df.groupby('route').agg(
    total_deliveries=('route', 'count'),
    avg_delivery_cost=('delivery_cost', 'mean')
).query("total_deliveries >= 20").sort_values(
    'avg_delivery_cost',
    ascending=False
)
print(most_expensive_route.head(10))





#Which routes experience the most delays?
#4. Which routes experience the most delays?
#Because is_delayed is:
#0 = Not delayed
#1 = Delayed
#the mean of is_delayed is actually the proportion of deliveries that were delayed.
route_analysis = df.groupby('route').agg(
    total_deliveries=('route', 'count'),
    avg_delivery_time=('delivery_time_hours', 'mean'),
    avg_delivery_cost=('delivery_cost', 'mean'),
    delay_rate=('is_delayed', 'mean')
).reset_index()

# Convert delay rate to percentage
route_analysis['delay_rate_pct'] = route_analysis['delay_rate'] * 100

#For example:
#0.65 = 65% delayed
route_analysis = route_analysis.reset_index()

most_delayed_route = route_analysis.loc[
    route_analysis['total_deliveries'] >= 20
].sort_values(
    'delay_rate',
    ascending=False
)

print(most_delayed_route.head(10))

route_analysis = route_analysis.reset_index()
#convert rate to %
route_analysis['delay_rate_pct'] = route_analysis['delay_rate'] * 100


# Filter and sort
most_delayed_route = route_analysis.loc[
    route_analysis['total_deliveries'] >= 20
].sort_values(
    'delay_rate_pct',
    ascending=False
)

print(most_delayed_route.head(10))

#5. I recommend adding route volume
#Which routes carry the most deliveries?
busiest_route = route_analysis.sort_values(
    'total_deliveries',
    ascending=False
)
busiest_route.head(10)

#This matters because a route with a high delay rate but only 20 deliveries 
# may be less operationally important than 
# a route with a slightly lower delay rate but 1,000 deliveries.


#6. Create a route performance table
route_performance = route_analysis[
    route_analysis['total_deliveries'] >= 20
].sort_values(
    'delay_rate_pct',
    ascending=False
)

route_performance.head(10)

route_performance[
    [
        'avg_delivery_cost',
        'avg_delivery_time',
        'delay_rate_pct',
        'total_deliveries'
    ]
].head(10)

#7. Now visualize the results
#Top 10 slowest routes
top_slowest = slowest_route.head(10).sort_values(
    'avg_delivery_time'
)

plt.figure(figsize=(10, 6))

plt.barh(
    top_slowest.index,
    top_slowest['avg_delivery_time']
)

plt.title('Top 10 Slowest Delivery Routes')
plt.xlabel('Average Delivery Time (Hours)')
plt.ylabel('Route')

plt.show()







#Top 10 most expensive routes
most_expensive_routes = route_analysis.loc[
    route_analysis['total_deliveries'] >= 20
].sort_values(
    'avg_delivery_cost',
    ascending=False
)
# Sort first, then take top 10
top_expensive = most_expensive_routes.sort_values(
    'avg_delivery_cost',
    ascending=False
).head(10)

plt.figure(figsize=(10, 6))

plt.barh(
    top_expensive['route'],   # use the route column
    top_expensive['avg_delivery_cost']
)

plt.title('Top 10 Most Expensive Delivery Routes')
plt.xlabel('Average Delivery Cost')
plt.ylabel('Route')

plt.show()


#Top 10 most delayed routes
most_delayed_routes = route_analysis.loc[
    route_analysis['total_deliveries'] >= 20
].sort_values(
    'delay_rate_pct',
    ascending=False
)
# Sort first, then take top 10
top_delayed = most_delayed_routes.sort_values(
    'delay_rate_pct',
    ascending=False
).head(10)

plt.figure(figsize=(10, 6))

plt.barh(
    top_delayed['route'],   # safer than using .index
    top_delayed['delay_rate_pct']
)

plt.title('Top 10 Routes by Delay Rate')
plt.xlabel('Delay Rate (%)')
plt.ylabel('Route')

plt.show()



#8. The most important part: interpret the results
#"Lagos → Kano has the highest delay rate."
#You want to investigate why.
#Lagos → Kano has the highest delay rate.
df[df['route'] == 'Lagos → Kano'].groupby(
    ['traffic_level', 'weather_condition']
).agg(
    avg_delivery_time = ('actual_delivery_time', 'mean'),
    delay_rate=('is_delayed', 'mean'),
    deliveries = ('route', 'size')
)
#Now you can potentially discover something like:
#The Lagos → Kano route experiences particularly high delays during high-traffic conditions.
#That is a business insight, not just an observation.


#Section 6: Operational KPIs (overall KPIs) says “Here’s how we’re doing.)
#These are metrics executives care about.
#1. Calculate the KPIs
#make the kpi table more meaningful
operational_kpis = pd.DataFrame({
    'KPI':[
        'Average Delivery Time',
        'Average Delivery Cost',
        'Delay Rate',
        'Average Customer Rating',
        'Average Cost Per KM',
        'Average Delivery Speed',
        'Total Deliveries'
    ],
    'Value':[
        round(df['actual_delivery_time'].mean(), 2),
        round(df['delivery_cost'].mean(), 2),
        round(df['is_delayed'].mean() * 100, 2),
        round(df['customer_rating'].mean(), 2),
        round(df['cost_per_km'].mean(), 2),
        round(df['delivery_speed_km_per_hr'].mean(), 2),
        df['order_id'].nunique()
    ],
    'Unit': [
        'Hours',
        'Currency',
        '%',
        'Rating / 5',
        'Currency / KM',
        'KM / Hours',
        'Deliveries'
    ]
})
operational_kpis

plt.figure(figsize=(10,6))
plt.barh(operational_kpis['KPI'], operational_kpis['Value'])
plt.xlabel('Value')
plt.title('Operational KPIs')
plt.show()
#section 7 KPI Breakdown by operational factors ((breakdowns) says “Here’s why we’re doing that way — and where to fix it.”)
#breakdown by vehicle
vehicle_kpis = df.groupby('vehicle_type').agg(
    avg_delivery_time=('actual_delivery_time', 'mean'),
    avg_delivery_cost=('delivery_cost', 'mean'),
    delay_rate=('is_delayed', 'mean'),
    avg_rating=('customer_rating', 'mean'),
    avg_cost_per_km=('cost_per_km', 'mean')
).round(2)

#breakdown by traffic
traffic_kpis = df.groupby('traffic_level').agg(
    avg_delivery_time=('actual_delivery_time', 'mean'),
    delay_rate=('is_delayed', 'mean'),
    avg_rating=('customer_rating', 'mean')
).round(2)

#breakdown by weather
weather_kpis = df.groupby('weather_condition').agg(
    avg_delivery_time=('actual_delivery_time', 'mean'),
    delay_rate=('is_delayed', 'mean'),
    avg_rating=('customer_rating', 'mean')
).round(2)

print(vehicle_kpis)
print(traffic_kpis)
print(weather_kpis)

#section 8
#Customer Satisfaction Analysis
#do delay reduce customer satisfaction
#customer rating by delay status 
rating_by_delay = df.groupby('is_delayed').agg(
    average_rating = ('customer_rating', 'mean'),
    number_of_deliveries = ('order_id', 'count')
).reset_index()

rating_by_delay['delay_status'] = rating_by_delay['is_delayed'].map({
    0: 'Not Delayed',
    1: 'Delayed'
})
#Remember:
#0 = Not delayed
#1 = Delayed
rating_by_delay

#visualize rating by delay status
rating_by_delay.plot(
    x='delay_status',
    y='average_rating',
    kind='bar',
    legend=False,
    figsize=(8,5)
)
plt.title('Average Customer Rating by Delay Status')
plt.xlabel('Delivery Status')
plt.ylabel('Average Customer Rating')
plt.xticks(rotation=0)
plt.show()


#How to interpret it
#If you find:
#Not Delayed → 4.2
#Delayed     → 3.4

#which vehicle keeps customers happiest
rating_by_vehicle =df.groupby('vehicle_type').agg(
    average_rating=('customer_rating', 'mean'),
    number_of_deliveries=('order_id', 'count'),
    delay_rate=('is_delayed','mean')
).round(2)
rating_by_vehicle
#sort by rating
rating_by_vehicle.sort_values(
    'average_rating',
    ascending=False
)
#The vehicle at the top has the highest average customer rating.

#visualize vehicle satisfaction
rating_by_vehicle['average_rating'].sort_values().plot(
    kind='barh',
    figsize=(8,5)
)
plt.title('Average Customer Rating by Vehicle Type')
plt.xlabel('Average Customer Rating')
plt.ylabel('Vehicle Type')
plt.show()

#does taffic affect customer satisfaction
rating_by_traffic =df.groupby('traffic_level').agg(
    average_rating=('customer_rating', 'mean'),
    number_of_deliveries=('order_id','count'),
    delay_rate=('is_delayed','mean')
).round(2)
rating_by_traffic
#sorting
rating_by_traffic.sort_values(
    'average_rating',
    ascending=False
)
#This allows you to see whether high traffic → more delays → lower ratings.

#customer rating by weather
rating_by_weather = df.groupby('weather_condition').agg(
    average_rating=('customer_rating','mean'),
    delay_rate=('is_delayed','mean')
).round(2)

rating_by_weather.sort_values(
    'average_rating',
    ascending=False
)
#Does customer satisfaction change under different weather conditions

#overall customer satisfaction table
customer_satisfaction_summary = pd.DataFrame({
    'Metric':[
        'Overall Average Rating',
        'Average Rating - Delayed',
        'Average Rating - Not Delayed',
        'Highest Rated Vehicle',
        'Lowest Rated Vehicle'
    ],
    'Value':[
        round(df['customer_rating'].mean(),2),
        round(df.loc[df['is_delayed'] == 1, 'customer_rating'].mean(), 2),
        round(df.loc[df['is_delayed'] == 0, 'customer_rating'].mean(), 2),
        rating_by_vehicle['average_rating'].idxmax(),
        rating_by_vehicle['average_rating'].idxmax()
    ]
})
customer_satisfaction_summary


#delay + vehicle + rating.
vehicle_delay_rating = df.groupby(
    ['vehicle_type', 'is_delayed']
).agg(
    average_rating=('customer_rating', 'mean'),
    deliveries=('order_id','count')
).round(2)

vehicle_delay_rating

#SECTION 8 — CUSTOMER SATISFACTION ANALYSIS
#8.1 Rating by Delay Status
   # → Do delayed deliveries have lower ratings?
#8.2 Rating by Vehicle Type
    #→ Which vehicle produces the highest satisfaction?
#8.3 Rating by Traffic Level
   # → Does congestion affect customer experience?
#8.4 Rating by Weather
    #→ Does weather affect satisfaction?
#8.5 Vehicle × Delay × Rating
   # → Does delay affect some vehicle types more than others?
#8.6 Business Insights
    #→ What does this mean for logistics operations?

#section 9
#Time-Based Analysis
#Orders by Month

monthly_orders = df.groupby('order_month').size().reset_index(name='total_orders')
monthly_orders

monthly_orders = monthly_orders.sort_values('order_month')
monthly_orders

#Make the month names easier to read
#Since order_month is numeric:
import calendar
monthly_orders['month_name'] = monthly_orders['order_month'].apply(
    lambda x: calendar.month_name[x]
)
monthly_orders

#arrange the columns
monthly_orders = monthly_orders[
    ['order_month', 'month_name', 'total_orders']
]
monthly_orders
#which month had the highest demand
highest_demand_month = monthly_orders.loc[
    monthly_orders['total_orders'].idxmax()
]
highest_demand_month

#get the month
monthly_orders.loc[
    monthly_orders['total_orders'].idxmax(),
    'month_name'
]

#month that recorded the highest delivery, suggesting that logistics capacity and vehicle availability should be closely monitored during this period.

#visualize monthly demand
plt.figure(figsize=(10, 5))
plt.plot(
    monthly_orders['month_name'],
    monthly_orders['total_orders'],
    marker='o'
)
plt.title('Monthly Delivery Demand')
plt.xlabel('Month')
plt.ylabel('Number of Deliveries')
plt.xticks(rotation=45)
plt.show()
#This lets you immediately see whether demand is:
#Increasing
#Decreasing
#Seasonal
#Relatively stable


#order by weekday
weekday_orders = df.groupby(
    'order_day_of_week'
).size().reset_index(name='total_order')
weekday_orders

day_orders = [
    'Sunday',
    'Monday',
    'Tuesday',
    'Wednesday',
    'Thursday',
    'Friday',
    'Saturday',
]
weekday_orders['order_day_of_week'] = pd.Categorical(
    weekday_orders['order_day_of_week'],
    categories=day_orders,
    ordered=True
)
weekday_orders = weekday_orders.sort_values(
    'order_day_of_week'
)
weekday_orders


#visualize orders by weekday
plt.figure(figsize=(10,5))
plt.bar(
    weekday_orders['order_day_of_week'],
    weekday_orders['total_order']
)
plt.title('Delivery Orders by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Number of Deliveries')
plt.xticks(rotation=45)
plt.show()
#identifying the busiest day.

#weekend Vs Weekday
df['day_type'] =df['is_weekend'].map({
    0: 'Weekday',
    1: 'Weekend'
})

#calculating the orders 
weekend_weekday_orders = df.groupby(
    'day_type'
).size().reset_index(name= 'total_orders')
weekend_weekday_orders


#Are Weekends Busier?
#i Don't just want to compare the number of orders.
#Because there are 5 weekdays but only 2 weekend days, total weekly orders will naturally tend to be higher on weekdays.
#A better comparison is:
#Average orders per day

daily_orders = df.groupby(
    ['order_date', 'day_type']
).size().reset_index(name= 'orders')

average_daily_orders = daily_orders.groupby(
    'day_type'
)['orders'].mean().reset_index()
average_daily_orders


#visualize weekend vs weekday
plt.figure(figsize=(7, 5))
plt.bar(
    average_daily_orders['day_type'],
    average_daily_orders['orders']
)
plt.title('Average Daily Delivery Volume: Weekday vs Weekday')
plt.xlabel('Day Type')
plt.ylabel('Average Orders per Day')
plt.show()

#monthly delays
#when do delay peak
monthly_delays = df.groupby('order_month').agg(
    total_orders=('order_id', 'count'),
    delayed_orders=('is_delayed', 'sum'),
    delay_rate=('is_delayed', 'mean')
).reset_index()
#%
import calendar
monthly_delays['delay_rate_pct'] = (
    monthly_delays['delay_rate'] * 100
).round(2)
#months
monthly_delays['month_name'] = monthly_delays['order_month'].apply(
    lambda x: calendar.month_name[x]
)

monthly_delays = monthly_delays[
    [
        'order_month',
        'month_name',
        'total_orders',
        'delayed_orders',
        'delay_rate_pct'
    ]
]
monthly_delays

#which month has the highest delay rate
peak_delay_month = monthly_delays.loc[
    monthly_delays['delay_rate_pct'].idxmax()
]
peak_delay_month

monthly_delays.loc[
    monthly_delays['delay_rate_pct'].idxmax(),
    'month_name'
]
monthly_delays['delay_rate_pct'].max()

#Visualize Monthly Delay Rate
plt.figure(figsize=(10, 5))

plt.plot(
    monthly_delays['month_name'],
    monthly_delays['delay_rate_pct'],
    marker='o'
)

plt.title('Monthly Delivery Delay Rate')
plt.xlabel('Month')
plt.ylabel('Delay Rate (%)')
plt.xticks(rotation=45)

plt.show()

#month, weather , vehicle with the highest delay rate, investigate why.
peak_month = monthly_delays.loc[
    monthly_delays['delay_rate_pct'].idxmax(),
    'order_month'
]

df[df['order_month'] == peak_month].groupby(
    'traffic_level'
).agg(
    delay_rate=('is_delayed', 'mean'),
    deliveries=('order_id', 'count')
).round(2)

df[df['order_month'] == peak_month].groupby(
    'weather_condition'
).agg(
    delay_rate=('is_delayed', 'mean'),
    deliveries=('order_id', 'count')
).round(2)

df[df['order_month'] == peak_month].groupby(
    'vehicle_type'
).agg(
    delay_rate=('is_delayed', 'mean'),
    deliveries=('order_id', 'count')
).round(2)

#SECTION 8 — TIME-BASED ANALYSIS

#8.1 Orders by Month
  #  ↓
#8.2 Orders by Weekday
  #  ↓
#8.3 Weekend vs Weekday
  #  ↓
#8.4 Monthly Delay Rate
  #  ↓
#8.5 Peak Demand Period
  #  ↓
#8.6 Peak Delay Period
   # ↓
#8.7 Root Cause of Peak Delays
   # ↓
#8.8 Business Insights


#Section 9: Business Insights
from IPython.display import display

print("===== OVERALL KPIs =====")
display(operational_kpis)

print("\n===== TOP ROUTES BY DELAY RATE =====")
display(most_delayed_routes.head(5))

print("\n===== TOP SLOWEST ROUTE =====")
display(slowest_route.head(5))

print("\n===== TOP EXPENSIVE ROUTE =====")
display(most_expensive_routes.head(5))

print("\n===== CUSTOMER RATING BY DELAY =====")
display(rating_by_vehicle.sort_values(
    'average_rating',
    ascending=False
))

print("\n===== MONTHLY ORDERS =====")
display(monthly_orders)

print("\n===== MONTHLY DELAY RATE =====")
display(monthly_delays)


#fixing customer rating 
rating_by_delay = df.groupby('is_delayed').agg(
    average_rating=('customer_rating', 'mean'),
    number_of_deliveries=('order_id','count')
).reset_index()

rating_by_delay['delay_status'] = rating_by_delay['is_delayed'].map({
    0: 'Not Delayed',
    1: 'Delayed'
})
display(
    rating_by_delay[
        ['delay_status', 'average_rating', 'number_of_deliveries']
    ]
)

#</> Markdown
# section 9 Business insights
#</> Markdown
##
## Business Insights

### Insight 1 — Delivery reliability is a major operational challenge

#The overall delivery delay rate is 66.49%, meaning approximately two-thirds of deliveries in the dataset were classified as delayed. 
# This indicates a significant opportunity to improve delivery reliability and operational efficiency.

### Insight 2 — Certain routes experience consistently high delay rates

#Lagos to Lagos recorded the highest route-level delay rate at 71.43%, followed by Ibadan to Port Harcourt at 70.42%. 
# These routes should be prioritized for further investigation and operational improvement.

### Insight 3 — Ibadan to Lagos is the slowest route

#Ibadan to Lagos recorded the highest average delivery time at approximately 7.24 hours. 
# This suggests that the route may require closer examination of travel conditions, scheduling, and route efficiency.

### Insight 4 — Delivery costs vary significantly across routes

#Abuja to Ibadan recorded the highest average delivery cost at approximately 386.75, followed by Lagos to Kano at approximately 380.67. 
# This indicates that route-level cost drivers should be investigated to identify opportunities for cost reduction.

### Insight 5 — Customer ratings are not meaningfully affected by delay status

#Average customer ratings were almost identical for delayed deliveries (3.7464) and non-delayed deliveries (3.7467). 
# Combined with the near-zero correlation between delay status and customer rating, this suggests that delays alone do not meaningfully explain differences in customer satisfaction within this dataset.

### Insight 6 — Delivery demand varies across months

#March recorded the highest monthly delivery volume with 672 orders, while February recorded the lowest with 572 orders. 
# Understanding these demand patterns can help improve resource and capacity planning.

### Insight 7 — Delivery reliability varies across months

#November recorded the highest monthly delay rate at 69.94%, while May recorded the lowest at 63.00%. 
# This indicates that operational performance varies throughout the year and may warrant further investigation into the factors contributing to peak-period delays.

### Insight 8 — Vehicle type has little effect on customer ratings

#Average customer ratings were very similar across vehicle types: 3.75 for bikes, 3.75 for trucks, and 3.74 for vans. 
# This suggests that vehicle type alone is not a major differentiator of customer satisfaction in this dataset.

# Section 10: Business Recommendations

##Recommendation 7 — Use vehicle type based on operational efficiency

##Your vehicle analysis showed very similar customer ratings across bikes, trucks and vans.

#Recommendation

##Vehicle allocation should therefore focus primarily on operational efficiency, delivery distance, cost per kilometer, and delivery requirements rather than customer ratings alone.

#Final Recommendation Summary

#You can finish the section with this:

## Recommendation Summary


#Based on the analysis, the logistics business should focus on improving delivery reliability, optimizing high-delay and high-cost routes, and strengthening capacity planning during periods of increased demand.


#Particular attention should be given to the Lagos to Lagos and Ibadan to Lagos routes due to their high delay rate and delivery time respectively. 
# High-cost routes such as Abuja to Ibadan should also be reviewed for potential cost optimization.


#The analysis also indicates that customer ratings are not meaningfully different between delayed and non-delayed deliveries. 
# Therefore, future analysis should investigate additional factors that may influence customer satisfaction.


#Overall, the findings demonstrate that route optimization, operational monitoring, capacity planning, and data-driven resource allocation can help improve logistics performance.


##DATA
 # ↓
##Cleaning
 # ↓
##Feature Engineering
  #↓
##Analysis
  #↓
##Findings
  #↓
##Business Insights
 # ↓
##Recommendations

######SECTION 11: Conclusion / Executive Summary
## Executive Summary

#This project analyzed 7,385 logistics delivery records to evaluate delivery performance, operational efficiency, customer satisfaction, route performance, and demand patterns.

#The analysis involved data cleaning, feature engineering, exploratory data analysis, correlation analysis, route-level analysis, operational KPI analysis, customer satisfaction analysis, and time-based analysis using Python and Pandas.

#The analysis identified a high overall delivery delay rate of 66.49%, highlighting delivery reliability as an important operational challenge. 
# At the route level, Lagos to Lagos recorded the highest delay rate at 71.43%, while Ibadan to Lagos recorded the longest average delivery time at approximately 7.24 hours.
# Abuja to Ibadan recorded the highest average delivery cost at approximately 386.75.

#Demand also varied across the year, with March recording the highest monthly delivery volume at 672 deliveries. 
# November recorded the highest monthly delay rate at 69.94%.

#Customer satisfaction analysis produced an interesting finding: average ratings for delayed and non-delayed deliveries were almost identical at 3.7464 and 3.7467 respectively. 
# This suggests that delivery delays alone do not meaningfully explain customer ratings in this dataset and that other factors may need to be investigated.

#Based on these findings, the key business recommendations are to prioritize high-delay routes for operational review, investigate high-cost and slow routes, improve capacity planning during high-demand periods, and identify additional factors that influence customer satisfaction.

#Overall, the analysis demonstrates how data can be used to identify operational inefficiencies, monitor logistics performance, and support evidence-based decision-making.

## Conclusion

#The analysis shows that the logistics operation has opportunities to improve delivery reliability, route efficiency, and cost management. 
#High-delay routes and high-cost routes should receive particular attention, while historical demand patterns can be used to improve resource planning.

#An important finding from the project is that customer ratings were not significantly different between delayed and non-delayed deliveries. 
#This demonstrates the importance of allowing the data to guide business conclusions rather than assuming that an expected relationship exists.

#Future analysis could investigate additional factors influencing customer satisfaction and explore predictive models for identifying deliveries that are likely to be delayed.


#Project Introduction
#Business Objectives
#Import Libraries
#Load Dataset
#Data Inspection
#Data Cleaning
#Feature Engineering
#Exploratory Data Analysis (EDA)
#Business Insights
#Recommendations
#Conclusion

#1. Introduction & Business Objectives
 #       ↓
#2. Data Loading
#        ↓
#3. Data Cleaning
 #       ↓
#4. Feature Engineering
  #      ↓
#5. Exploratory Data Analysis
 #       ↓
#6. Correlation Analysis
 #       ↓
#7. Route Analysis
 #       ↓
#8. Operational KPIs
 #       ↓
#9. Customer / Time Analysis
 #       ↓
#10. Business Insights
 #       ↓
#11. Recommendations
 #       ↓
#12. Conclusion
#THE END
# TO GET MY UPDATED DATABASE 

# Define the folder path where you want to save
import os

# Make sure the folder exists
os.makedirs("data/processed", exist_ok=True)

# Save your dataset
df.to_csv("data/processed/cleaned_logistics_data.csv", index=False)

