# LOGISTICS DELIVERY PERFORMANCE ANALYSIS

A Python-Based Data Cleaning, Feature Engineering and Business Analytics Project 
- Tools: Python, Pandas, NumPy, Matplotlib, VS Code  
                       **Portfolio Project Report**



```
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
```
```
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
```


- ## Table of Contents
#### 1. [Executive Summary](https://github.com/victorhamvida-dotcom/LOGISTICS-DELIVERY-PERFORMANCE-ANALYSIS#1-executive-summary)

### 2. [Business Problem](https://github.com/victorhamvida-dotcom/LOGISTICS-DELIVERY-PERFORMANCE-ANALYSIS#2-business-problem)  
  ###### 2.1 [Business Objectives](https://github.com/victorhamvida-dotcom/LOGISTICS-DELIVERY-PERFORMANCE-ANALYSIS#21-business-objectives)

### 3. [Dataset Overview](https://github.com/victorhamvida-dotcom/LOGISTICS-DELIVERY-PERFORMANCE-ANALYSIS#3-dataset-overview)

### 4. [Data Cleaning](https://github.com/victorhamvida-dotcom/LOGISTICS-DELIVERY-PERFORMANCE-ANALYSIS#4-data-cleaning)

### 5. [Feature Engineering](https://github.com/victorhamvida-dotcom/LOGISTICS-DELIVERY-PERFORMANCE-ANALYSIS#5-feature-engineering)

### 6. [Exploratory and Correlation Analysis](https://github.com/victorhamvida-dotcom/LOGISTICS-DELIVERY-PERFORMANCE-ANALYSIS#6-exploratory-and-correlation-analysis)
  ###### 6.1 [Distance and Delivery Time](https://github.com/victorhamvida-dotcom/LOGISTICS-DELIVERY-PERFORMANCE-ANALYSIS#61-distance-and-delivery-time)
  ###### 6.2 [Distance and Delivery Cost](https://github.com/victorhamvida-dotcom/LOGISTICS-DELIVERY-PERFORMANCE-ANALYSIS#62-distance-and-delivery-cost)
  ###### 6.3 [Delay and Customer Rating](https://github.com/victorhamvida-dotcom/LOGISTICS-DELIVERY-PERFORMANCE-ANALYSIS#63-delay-and-customer-rating)

### 7. [Route Performance Analysis](https://github.com/victorhamvida-dotcom/LOGISTICS-DELIVERY-PERFORMANCE-ANALYSIS#7-route-performance-analysis) 
  ###### 7.1 [Routes with the Highest Delay Rates](https://github.com/victorhamvida-dotcom/LOGISTICS-DELIVERY-PERFORMANCE-ANALYSIS#71-routes-with-the-highest-delay-rates)
  ###### 7.2 [Slowest Routes](https://github.com/victorhamvida-dotcom/LOGISTICS-DELIVERY-PERFORMANCE-ANALYSIS#72-slowest-routes)
  ###### 7.3 [Most Expensive Routes](https://github.com/victorhamvida-dotcom/LOGISTICS-DELIVERY-PERFORMANCE-ANALYSIS#73-most-expensive-routes)

### 8. [Operational KPI Dashboard](https://github.com/victorhamvida-dotcom/LOGISTICS-DELIVERY-PERFORMANCE-ANALYSIS#8-operational-kpi-dashboard)

### 9. [Customer Satisfaction Analysis](https://github.com/victorhamvida-dotcom/LOGISTICS-DELIVERY-PERFORMANCE-ANALYSIS#9-customer-satisfaction-analysis) 
  ###### 9.1 [Customer Rating by Delay Status](https://github.com/victorhamvida-dotcom/LOGISTICS-DELIVERY-PERFORMANCE-ANALYSIS#91-customer-rating-by-delay-status)
  ###### 9.2 [Customer Rating by Vehicle Type](https://github.com/victorhamvida-dotcom/LOGISTICS-DELIVERY-PERFORMANCE-ANALYSIS#92-customer-rating-by-vehicle-type)
  ###### 9.3 [Weather and Delivery Performance](https://github.com/victorhamvida-dotcom/LOGISTICS-DELIVERY-PERFORMANCE-ANALYSIS#93-weather-and-delivery-performance)
  ###### 9.4 [Traffic Conditions](https://github.com/victorhamvida-dotcom/LOGISTICS-DELIVERY-PERFORMANCE-ANALYSIS#94-traffic-conditions)

### 10. [Time-Based Analysis](https://github.com/victorhamvida-dotcom/LOGISTICS-DELIVERY-PERFORMANCE-ANALYSIS#10-time-based-analysis)
  ###### 10.1 [Monthly Delivery Volume](https://github.com/victorhamvida-dotcom/LOGISTICS-DELIVERY-PERFORMANCE-ANALYSIS#101-monthly-delivery-volume)
  ###### 10.2 [Monthly Delay Rate](https://github.com/victorhamvida-dotcom/LOGISTICS-DELIVERY-PERFORMANCE-ANALYSIS#102-monthly-delay-rate)

### 11. [Key Business Insights](https://github.com/victorhamvida-dotcom/LOGISTICS-DELIVERY-PERFORMANCE-ANALYSIS#11-key-business-insights)

### 12. [Business Recommendations](https://github.com/victorhamvida-dotcom/LOGISTICS-DELIVERY-PERFORMANCE-ANALYSIS#12-business-recommendations)

### 13. [Conclusion](https://github.com/victorhamvida-dotcom/LOGISTICS-DELIVERY-PERFORMANCE-ANALYSIS#13-conclusion)

### 14. [Technical Skills Demonstrated](https://github.com/victorhamvida-dotcom/LOGISTICS-DELIVERY-PERFORMANCE-ANALYSIS#14-technical-skills-demonstrated)

### 15. [Project Outcome](https://github.com/victorhamvida-dotcom/LOGISTICS-DELIVERY-PERFORMANCE-ANALYSIS#15-project-outcome)

### 16. [Appendix A: Core Analytical Questions](https://github.com/victorhamvida-dotcom/LOGISTICS-DELIVERY-PERFORMANCE-ANALYSIS#appendix-a-core-analytical-questions)




## 1. Executive Summary
This project analyzes logistics delivery records to evaluate delivery performance, operational efficiency, route performance, delivery costs, customer satisfaction, and demand patterns.

The analysis was conducted using Python, with Pandas and NumPy used for data manipulation and analysis, while Matplotlib and Seaborn were used for data visualization.

The analysis identified an overall delivery delay rate of **66.49%**, an average delivery time of **36.60 hours**, an average delivery cost of **359.08**, and an average customer rating of **3.75 out of 5**.

Route-level analysis showed that **Lagos to Lagos recorded the highest delay rate at 71.43%**, while **Ibadan to Lagos recorded the highest average delivery time at approximately 7.24 hours** among the routes examined. **Abuja to Ibadan** recorded the highest average delivery cost at approximately **386.75**.

The analysis also found a strong positive relationship between distance and delivery time, as well as between distance and delivery cost. However, there was almost no relationship between delivery delays and customer ratings.

Based on these findings, recommendations were developed around route optimization, delivery reliability, cost management, capacity planning, and further investigation of customer satisfaction drivers.

## 2. Business Problem
Logistics companies need to maintain reliable delivery times while controlling transportation costs and maintaining customer satisfaction. Operational inefficiencies such as delivery delays, expensive routes, and variations in delivery performance can negatively affect profitability and service quality.

The purpose of this analysis is to examine logistics delivery data and identify patterns in delivery performance, route efficiency, operational costs, delays, and customer satisfaction.

The analysis is designed to help management understand where operational problems occur and identify areas where improvements may be required.
The objective of the project was to use delivery data to identify operational inefficiencies and provide evidence-based recommendations that could support logistics management.
- 	Measure overall delivery performance using operational KPIs.
- 	Identify routes with high delay rates, long delivery times and high delivery costs.
- 	Understand the relationship between distance, delivery time and delivery cost.
- 	Evaluate customer satisfaction across delay status, vehicle type, traffic and weather conditions.
- 	Identify monthly demand and delay patterns.
- 	Translate analytical findings into practical business recommendations.

## 2.1 Business Objectives
The key objectives of the analysis were to:
- 	Evaluate overall delivery performance using operational KPIs. 
- 	Identify routes with high delay rates. 
- 	Identify the slowest delivery routes. 
- 	Identify the most expensive routes. 
- 	Examine relationships between distance, delivery time, and delivery cost. 
- 	Analyze customer ratings by delay status, vehicle type, traffic, and weather. 
- 	Identify monthly delivery demand and delay patterns. 
- 	Develop meaningful operational KPIs. 
- 	Generate actionable business insights. 
- 	Provide data-driven recommendations for improving logistics operations.
  
## 3. Dataset Overview
The dataset contains logistics delivery records representing orders, delivery characteristics, operational conditions, and customer feedback.

The dataset was designed to represent a realistic logistics environment and contained several data quality issues that required cleaning and transformation before analysis.

The project began with a deliberately dirty logistics dataset containing inconsistent values, missing data and duplicate order identifiers. The data covered delivery orders involving five cities and multiple vehicle, traffic and weather categories.

| Dataset                                   | Metric/Result                                      |
|-------------------------------------------|----------------------------------------------------|
| Records after cleaning                     | 7,385                                              |
| Unique order IDs                           | 7,383                                              |
| Original variables                         | 16                                                 |
| Final variables after feature engineering  | 39 during analysis                                 |
| Origin/destination cities                  | 5 cities                                           |
| Vehicle types                              | 3 (Bike, Truck, Van)                               |
| Weather categories                         | 3 (Clear, Rain, Storm)                             |
| Traffic categories                         | 3 (Low, Medium, High)                              |
| Main Analysis Areas                        | Delivery performance, routes, cost, delays, customer satisfaction |

The distinction between total records and unique order IDs was retained during the analysis to ensure that the dataset structure was accurately documented. This distinction is documented rather than silently treating the two measures as identical.

## 4. Data Cleaning
Data cleaning was performed before analysis to improve consistency and reliability. The original dataset contained 8,160 rows and several quality issues.

-	Standardized inconsistent city names such as 'lagos'/'Lagos' and 'port harcourt'/'Port Harcourt'.
-	Standardized vehicle labels such as 'BIKE', 'Bike', 'VAN', 'Van', 'TRUCK' and 'Truck'.
-	Converted order and delivery dates into datetime format.
-	Investigated and handled missing distance, delivery cost and customer rating values.
-	Identified duplicate order IDs and reviewed duplicate records.
-	Validated the cleaned dataset using null-value checks and data-type inspection.
-	Converted appropriate categorical fields to category dtype for cleaner analysis.
  
After cleaning, the analytical dataset contained no remaining missing values in the fields used for the final analysis.

The original logistics dataset required several data cleaning procedures before it could be used for analysis. The objective of the cleaning process was to improve data consistency, ensure appropriate data types, handle data quality issues, and prepare the dataset for reliable analysis. Eg**Missing Values**, **Duplicate Records**, **Data Type Conversion**, **Categorical Standardization**, **Data Validation**

After cleaning, the dataset was validated by checking missing values, data types, duplicate records, and the resulting dataset structure.

The cleaned dataset was then used for feature engineering and subsequent analysis.

## 5. Feature Engineering
Feature engineering was performed to transform the raw logistics data into variables that could provide greater business value during analysis.

Several new features were created to support operational, route, cost, customer satisfaction, and time-based analysis.

| Feature                     | Purpose                                           | Business Use                                                                 |
|------------------------------|---------------------------------------------------|-------------------------------------------------------------------------------|
| Actual delivery time         | Measures elapsed time between order and delivery. | Monitor delivery speed.                                                       |
| Delay hours                  | Quantifies the amount of delay.                   | Measure delivery reliability.                                                 |
| Route                        | Combines origin and destination.                  | Compare route performance.                                                    |
| Cost per KM                  | Normalizes cost by distance.                      | Compare cost efficiency.                                                      |
| Delivery speed               | Distance divided by delivery time.                | Assess operational efficiency.                                                |
| Same-city delivery           | Flags deliveries where origin equals destination. | Compare local vs inter-city operations.                                       |
| Month / weekday / weekend    | Extracts calendar patterns.                       | Support capacity planning.                                                    |
| Distance category            | Groups deliveries by distance.                    | Compare short, medium, and long trips; assess profitability.                  |
| Daily order volume / peak day| Measures daily demand pressure.                   | Support staffing and capacity planning.                                       |
| Is Delayed                   | Identifies whether a delivery was delayed.        | Spot delays that can affect the business.                                     |
| Order Month                  | Supports monthly demand analysis.                 | Track business growth month by month.                                         |
| Weekday                      | Supports weekday demand analysis.                 | Track business growth week by week.                                           |
| Weekend/Weekday              | Enables comparison of weekend and weekday ops.    | Managers can plan improvements based on outcomes.                             |


These engineered variables allowed the analysis to move beyond the original raw fields and answer more practical business questions.

## 6. Exploratory and Correlation Analysis
Correlation analysis was conducted to identify relationships between numerical logistics variables.

The analysis focused particularly on the relationships between distance, delivery time, delivery cost, delay status, and customer rating.

Correlation analysis showed that distance was strongly associated with delivery time and delivery cost. This is operationally intuitive: longer shipments generally require more time and incur greater cost.

![](https://github.com/victorhamvida-dotcom/LOGISTICS-DELIVERY-PERFORMANCE-ANALYSIS/blob/main/correlation%20matric%20of%20logistics%20variables%202.png)

![](https://github.com/victorhamvida-dotcom/LOGISTICS-DELIVERY-PERFORMANCE-ANALYSIS/blob/main/key%20corelation%20identified.png)


 
Figure 1. Selected correlations heat map from  the completed analysis.

| Relationship                 | Correlation |
|-------------------------------|-------------|
| Distance vs Delivery Time     | 0.817       |
| Distance vs Delivery Cost     | 0.732       |
| Delivery Time vs Cost         | 0.649       |
| Delay vs Customer Rating      | 0.002       |


The relationship between delay status and customer rating was effectively zero (r = 0.0019). Therefore, the analysis does not support the assumption that delayed deliveries alone explain lower ratings.

### 6.1 Distance and Delivery Time

Distance showed a strong positive relationship with delivery time. This indicates that longer delivery distances generally require more time to complete.

The correlation between distance and delivery time was approximately **0.82**, representing a strong positive relationship.

### 6.2 Distance and Delivery Cost
Distance also showed a strong positive relationship with delivery cost.

The correlation between distance and delivery cost was approximately **0.73**, indicating that longer-distance deliveries generally tend to incur higher delivery costs.

### 6.3 Delay and Customer Rating
The relationship between delivery delay and customer rating was almost zero, with a correlation of approximately **0.002**.

This suggests that delays alone do not meaningfully explain differences in customer ratings within this dataset.

This is an important finding because it demonstrates that business assumptions should be validated using actual data rather than assumed relationships.

## 7. Route Performance Analysis
Route analysis focused on three operational questions: which routes experience the most delays, which are slowest, and which are most expensive.

### 7.1 Routes with the Highest Delay Rates
 
Figure 2. Ten routes with the highest delay rates.
![](https://github.com/victorhamvida-dotcom/LOGISTICS-DELIVERY-PERFORMANCE-ANALYSIS/blob/main/top%2010%20route%20by%20delay%20rate.png)

Five routes with the highest delay rates.
![](https://github.com/victorhamvida-dotcom/LOGISTICS-DELIVERY-PERFORMANCE-ANALYSIS/blob/main/top%205%20routes%20by%20delay%20rates.png)

Lagos to Lagos recorded the highest route-level delay rate at **71.43%**, followed by Ibadan to Port Harcourt at **70.42%**.

Other routes among the five highest-delay routes included Lagos to Ibadan, Ibadan to Lagos, and Kano to Port Harcourt.

The high delay rates observed across these routes suggest that they should be prioritized for further operational investigation.

Management could examine dispatch scheduling, route planning, traffic conditions, vehicle allocation, and delivery scheduling to determine the factors contributing to delays.

### 7.2 Slowest Routes
 
Figure 3. Ten slowest routes by average delivery time.
![](https://github.com/victorhamvida-dotcom/LOGISTICS-DELIVERY-PERFORMANCE-ANALYSIS/blob/main/top%2010%20slowest%20delivery%20route.png)

 Five slowest routes by average delivery time.
![](https://github.com/victorhamvida-dotcom/LOGISTICS-DELIVERY-PERFORMANCE-ANALYSIS/blob/main/top%205%20slowest%20routes.png)
 

Ibadan to Lagos recorded the highest average delivery time among the routes examined, at approximately **7.24 hours**.

This was followed by Abuja to Port Harcourt at approximately **7.01 hours** and Ibadan to Port Harcourt at approximately **6.99 hours**.

The performance of these routes should be investigated to determine whether route planning, scheduling, traffic conditions, or other operational factors are contributing to longer delivery times.

### 7.3 Most Expensive Routes
 
Figure 4. Ten most expensive routes by average delivery cost.
 ![](https://github.com/victorhamvida-dotcom/LOGISTICS-DELIVERY-PERFORMANCE-ANALYSIS/blob/main/top%2010%20most%20expensive%20delivery%20route.png)

Five most expensive routes by average delivery cost.
![](https://github.com/victorhamvida-dotcom/LOGISTICS-DELIVERY-PERFORMANCE-ANALYSIS/blob/main/top%205%20most%20expensive%20routes.png)

Abuja to Ibadan recorded the highest average delivery cost at approximately **386.75**.

Other high-cost routes included Lagos to Kano, Lagos to Lagos, Kano to Ibadan, and Abuja to Port Harcourt.

These routes may provide opportunities for cost optimization. Management should investigate factors such as distance, vehicle allocation, route efficiency, and other operational costs associated with these deliveries.

## 8. Operational KPI Dashboard
| KPI                     | Value  | Unit          |
|--------------------------|--------|---------------|
| Average Delivery Time    | 36.60  | Hours         |
| Average Delivery Cost    | 359.08 | Currency      |
| Delay Rate               | 66.49  | %             |
| Average Customer Rating  | 3.75   | Rating / 5    |
| Average Cost per KM      | 1.62   | Currency / KM |
| Average Delivery Speed   | 16.20  | KM / Hour     |
| Total Delivery Records   | 7,385  | Records       |


The most significant KPI finding was the **66.49%** overall delay rate.

This indicates that approximately two-thirds of the delivery records were classified as delayed, highlighting delivery reliability as a major area for operational improvement.

The average delivery cost was **359.08**, while the average cost per kilometer was **1.62**.

The average customer rating of **3.75 out of 5** indicates a moderate level of customer satisfaction within the dataset.

The 66.49% delay rate is the most prominent operational KPI finding and represents the clearest area for performance improvement.

## 9. Customer Satisfaction Analysis
Customer satisfaction was analyzed using customer ratings across different operational factors.

The analysis focused on delay status, vehicle type, traffic conditions, and weather conditions.

![](https://github.com/victorhamvida-dotcom/LOGISTICS-DELIVERY-PERFORMANCE-ANALYSIS/blob/main/operational%20performance%20by%20vehicle%20type.png)

Vehicle-level delivery performance.

### 9.1 Customer Rating by Delay Status

The analysis produced the following results:

| Delivery Status | Average Rating | Number of Deliveries |
|-----------------|----------------|----------------------|
| Not Delayed     | 3.7467         | 2,475                |
| Delayed         | 3.7464         | 4,910                |


 ![](https://github.com/victorhamvida-dotcom/LOGISTICS-DELIVERY-PERFORMANCE-ANALYSIS/blob/main/average%20customer%20rating%20by%20delay%20status.png)


  ![](https://github.com/victorhamvida-dotcom/LOGISTICS-DELIVERY-PERFORMANCE-ANALYSIS/blob/main/delayed%20deliveries%20distribution.png)



  
Figure 5. Delay rate by delay status.

Customer ratings were almost identical between delayed and non-delayed deliveries.

Delayed deliveries recorded an average rating of **3.7464**, compared with **3.7467** for non-delayed deliveries.

The difference is extremely small, suggesting that delivery delays alone do not have a meaningful effect on customer ratings within this dataset.

This finding is also consistent with the near-zero correlation between delay status and customer rating.

Therefore, further investigation would be required to determine which other factors influence customer satisfaction.


### 9.2 Customer Rating by Vehicle Type
 


  ![](https://github.com/victorhamvida-dotcom/LOGISTICS-DELIVERY-PERFORMANCE-ANALYSIS/blob/main/average%20customer%20rating%20by%20vehicle%20type.png)
Figure 6. Average Delay rate by vehicle type.

  ![](https://github.com/victorhamvida-dotcom/LOGISTICS-DELIVERY-PERFORMANCE-ANALYSIS/blob/main/customer%20rating%20by%20vehicle%20type.png)
 

Figure 7. Delay rate by vehicle type.


Average customer ratings were very similar across vehicle types.

Bikes recorded an average rating of approximately **3.75**, trucks also recorded approximately **3.75**, while vans recorded approximately **3.74**.

This suggests that vehicle type alone is not a major differentiator of customer satisfaction in this dataset.

Vehicle selection should therefore focus on operational efficiency, delivery distance, cost, and delivery requirements rather than customer rating alone.


### 9.3 Weather and Delivery Performance
 
![](https://github.com/victorhamvida-dotcom/LOGISTICS-DELIVERY-PERFORMANCE-ANALYSIS/blob/main/delay%20rate%20by%20weather%20condition.png)
Figure 8. Delay rate by weather condition.

Weather conditions showed some variation in delivery performance.

Storm conditions recorded the highest delay rate at approximately **68%**, compared with **66% during rain** and **65% under clear conditions**.

Storm conditions also recorded the lowest average customer rating among the weather categories examined.

This suggests that adverse weather conditions may be associated with poorer delivery performance and should be considered when planning delivery schedules and operational resources.

### 9.4 Traffic conditions


 ![](https://github.com/victorhamvida-dotcom/LOGISTICS-DELIVERY-PERFORMANCE-ANALYSIS/blob/main/delay%20rate%20by%20traffic%20level.png)
Figure 9. Delay rate by traffic conditions.


## 10. Time-Based Analysis
Time-based analysis was conducted to identify patterns in delivery demand and delivery delays across different periods.

### 10.1 Monthly Delivery Volume

 
Figure 10. Monthly delivery volume.

![](https://github.com/victorhamvida-dotcom/LOGISTICS-DELIVERY-PERFORMANCE-ANALYSIS/blob/main/monthly%20delivery%20demand.png)
 

![](https://github.com/victorhamvida-dotcom/LOGISTICS-DELIVERY-PERFORMANCE-ANALYSIS/blob/main/monthly%20delay%20rate.png)

Delivery demand varied across the year.

**March recorded the highest monthly delivery volume with 672 orders**, while February recorded the lowest volume with **572 orders**.

Understanding these demand patterns can help logistics management plan driver availability, vehicle capacity, staffing, and other operational resources.


### 10.2 Monthly Delay Rate
 
Figure 11. Monthly delay rate compared with the overall delay rate.
![](https://github.com/victorhamvida-dotcom/LOGISTICS-DELIVERY-PERFORMANCE-ANALYSIS/blob/main/monthly%20delivery%20delay%20rate.png)
 
![](https://github.com/victorhamvida-dotcom/LOGISTICS-DELIVERY-PERFORMANCE-ANALYSIS/blob/main/monthly%20delay%20rate%202.png)
 




Monthly delay rates also varied throughout the year.

**November recorded the highest monthly delay rate at 69.94%**, followed by October at **69.44%**.

May recorded the lowest monthly delay rate at **63.00%**.

The variation suggests that operational performance changes across different periods and should be monitored to determine whether additional resources or operational adjustments are required during periods of higher delay rates.

## 11. Key Business Insights

The analysis produced several important business insights.

- **Insight 1: Delivery Reliability Is a Major Operational Challenge**
The overall delivery delay rate was **66.49%**, indicating that delivery reliability is a significant operational concern.

- **Insight 2: Certain Routes Experience High Delay Rates**
Lagos to Lagos recorded the highest route-level delay rate at **71.43%**, followed by Ibadan to Port Harcourt at **70.42%**.
These routes should receive additional operational attention.

- **Insight 3: Ibadan to Lagos Is the Slowest Route**
Ibadan to Lagos recorded the highest average delivery time among the routes examined at approximately **7.24 hours**.
This indicates a potential opportunity for route and scheduling optimization.

- **Insight 4: Abuja to Ibadan Has the Highest Average Delivery Cost**
Abuja to Ibadan recorded the highest average delivery cost at approximately **386.75**.
The route should be investigated to understand the factors contributing to its relatively high cost.

- **Insight 5: Delays Do Not Meaningfully Explain Customer Ratings**
Customer ratings for delayed and non-delayed deliveries were almost identical.
The average rating was **3.7464 for delayed deliveries** compared with **3.7467 for non-delayed deliveries**.
This suggests that other factors may play a more important role in determining customer satisfaction.

- **Insight 6: Delivery Demand Varies by Month**
March recorded the highest monthly delivery volume with **672 orders**.
This provides an opportunity to use historical demand patterns to improve capacity planning.

- **Insight 7: November Experienced the Highest Monthly Delay Rate**
November recorded the highest monthly delay rate at **69.94%**, suggesting that this period may require further operational investigation.

- **Insight 8: Weather Conditions May Influence Delivery Reliability**
Storm conditions recorded the highest delay rate among the weather categories examined.
This suggests that weather conditions should be considered when planning delivery operations and contingency measures.

## 12. Business Recommendations
Based on the findings from the analysis, the following recommendations are proposed.

- **Recommendation 1: Prioritize High-Delay Routes**
Management should prioritize routes with consistently high delay rates for operational review.

Particular attention should be given to Lagos to Lagos and Ibadan to Port Harcourt.
The review should examine dispatch scheduling, route planning, traffic conditions, vehicle allocation, and delivery scheduling.

- **Recommendation 2: Investigate Slow Delivery Routes**
The Ibadan to Lagos route recorded the highest average delivery time among the routes examined.
Management should investigate the operational factors contributing to the longer delivery time and identify opportunities for route and scheduling optimization.

- **Recommendation 3: Review High-Cost Routes**
High-cost routes such as Abuja to Ibadan should be reviewed to identify the main cost drivers.
Management should evaluate vehicle allocation, route efficiency, distance, and other operational factors that may contribute to higher delivery costs.

- **Recommendation 4: Improve Capacity Planning**
Historical monthly demand patterns should be used to improve resource planning.
Periods with higher delivery volumes may require additional drivers, vehicles, or operational capacity to prevent service deterioration.

- **Recommendation 5: Prepare for Adverse Weather**
Because storm conditions recorded a higher delay rate than clear and rainy conditions, logistics operations should incorporate weather-related contingency planning.
Delivery schedules and operational resources can be adjusted when adverse weather conditions are expected.

- **Recommendation 6: Investigate Other Drivers of Customer Satisfaction**
The analysis showed almost no difference between ratings for delayed and non-delayed deliveries.
Therefore, management should investigate other potential drivers of customer satisfaction, including traffic conditions, weather, vehicle type, delivery experience, and service quality.

- **Recommendation 7: Establish Continuous KPI Monitoring**
The company should continuously monitor key operational metrics such as:
-	Delay rate 
-	Average delivery time 
-	Average delivery cost 
-	Cost per kilometer 
-	Delivery speed 
-	Customer rating 
-	Route performance 
Regular monitoring would allow management to identify performance deterioration early and take corrective action.

## 13. Conclusion
This project demonstrates how Python based data analysis can be used to transform logistics delivery data into actionable business insights.

Through data cleaning, feature engineering, exploratory data analysis, correlation analysis, route analysis, KPI development, customer satisfaction analysis, and time-based analysis, the project identified important opportunities to improve delivery reliability, route efficiency, cost management, and operational planning.

The most significant finding was the **66.49% overall delivery delay rate**, indicating that delivery reliability represents a major operational challenge.

The route analysis also identified specific routes with high delay rates, long delivery times, and high delivery costs. These findings provide management with clear areas for further investigation and potential optimization.

An important finding from the customer satisfaction analysis was that delayed and non-delayed deliveries had almost identical customer ratings. This demonstrates the importance of allowing the data to guide business conclusions rather than assuming that a particular operational factor will automatically influence customer satisfaction.

Overall, the project demonstrates the ability to move from **raw data → data cleaning → feature engineering → analysis → business insights → recommendations**, providing a complete data analytics workflow.

Future analysis could extend this project into predictive analytics by developing a model to identify deliveries at high risk of delay before dispatch. An interactive business intelligence dashboard could also be developed to allow management to monitor logistics KPIs and route performance on an ongoing basis.

## 14. Technical Skills Demonstrated
This project demonstrates practical experience in the following areas:
-	Python programming 
-	Pandas 
-	NumPy 
-	Matplotlib 
-	Seaborn 
-	Data cleaning 
-	Data validation 
-	Missing-value handling 
-	Duplicate identification 
-	Data type conversion 
-	Categorical data standardization 
-	Feature engineering 
-	Exploratory Data Analysis (EDA) 
-	Correlation analysis 
-	Data visualization 
-	KPI development 
-	Route performance analysis 
-	Customer satisfaction analysis 
-	Time-series analysis 
-	Business insight generation 
-	Business recommendations
  
## 15. Project Outcome
The project successfully transformed a raw and imperfect logistics dataset into a structured analytical dataset and used it to identify operational performance patterns.

The analysis demonstrated that data can be used to identify high-risk routes, measure delivery efficiency, understand cost patterns, evaluate customer satisfaction, and support operational decision-making.

The final recommendations provide practical areas for management to investigate, particularly around delivery delays, route optimization, cost efficiency, capacity planning, and customer experience.

## Appendix A: Core Analytical Questions
-	Which routes have the highest delay rates?
-	Which routes are the slowest?
-	Which routes are the most expensive?
-	How strongly are distance, delivery time and delivery cost related?
-	Does delay status meaningfully affect customer ratings?
-	Which vehicle types perform best operationally?
-	How do traffic and weather conditions relate to delivery performance?
-	Which months have the highest demand and delay rates?
 
![](https://github.com/victorhamvida-dotcom/LOGISTICS-DELIVERY-PERFORMANCE-ANALYSIS/blob/main/code%20sinppet.png)

![](https://github.com/victorhamvida-dotcom/LOGISTICS-DELIVERY-PERFORMANCE-ANALYSIS/blob/main/flow%20chart.png)
