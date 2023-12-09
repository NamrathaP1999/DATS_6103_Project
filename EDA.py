#%%
crime_data.to_csv('cleaned_data.csv',index = False)

#%%
# II) Exploratory Data Analysis 

#%%
# required for data analysis and modeling tasks.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import LabelEncoder 
from scipy.stats import chi2_contingency
from plotly.subplots import make_subplots

#%%
crime_df = pd.read_csv("cleaned_data.csv", encoding='latin1')
crime_df.shape
crime_df.columns

#%%
crime_df['OCCURRED_ON_DATE'] = pd.to_datetime(crime_df['OCCURRED_ON_DATE'])
crime_df['DATE'] = crime_df['OCCURRED_ON_DATE'].dt.day
#%%
# Dropping unwanted rows
crime_df.drop(['INCIDENT_NUMBER','OFFENSE_DESCRIPTION','OCCURRED_ON_DATE','OFFENSE_CODE','Location'], axis=1, inplace = True)

#%%
crime_df.head()
#%%
crime_df.describe().T

# %%[markdown]
#UNIVARIATE ANALYSIS
#%%
#Count plot for YEAR
sns.countplot(x = crime_df['YEAR'], color = '#FAB4C6')
plt.title('Distribution of Incidents by Year')
plt.xticks(rotation=90)
plt.show()

#%%[markdown]
# The incidents were highest in the year 2017 and lowest in the year 2015
#%%
#Count plot for MONTH
sns.countplot(x = crime_df['MONTH'], color = '#C8E4C5')
plt.title('Distribution of Incidents by Month')
plt.xticks(rotation=90)
plt.show()

#%%[markdown]
# The incidents were highest in the month of July and August
#%%
#Count plot for HOUR
sns.countplot(x = crime_df['HOUR'], color = '#FFB347')
plt.title('Distribution of Incidents by Hour')
plt.xticks(rotation=90)
plt.show()

#%%[markdown]
# The Incidents recorded are higher between 9th hour to 19th hour and lower during the early morning
#%%
#Count plot for DATE
sns.countplot(x = crime_df['DATE'], color = '#E6E200')
plt.title('Distribution of Incidents by Day')
plt.xticks(rotation=90)
plt.show()
#%%[markdown]
# The Incidents recorded are more during the beginning and the mid of the month. 31st of the month recored the lowest among all days. 
#%%
#Count plot for DAY_OF_WEEK
days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

sns.countplot(x = crime_df['DAY_OF_WEEK'], order = days_order, color = '#769FB6')

plt.title('Distribution of Incidents by Day of the Week')
plt.xticks(rotation=45)
plt.show()
#%%[markdown]
# It can be observed that the Incident rate is almost equal on all the days except the highest being on Friday and lowest on Sunday. 

#%%
sns.countplot(x = crime_df['DISTRICT'], color = '#CDB5CD')
plt.title('Distribution of Incidents by District')
plt.xticks(rotation=90)
plt.show()

#%%[markdown]
# Description for DISTRICT
#%%[markdown]
# Analysis for SHOOTING variable
#%%[markdown]
# 1) Countplot for shooting
sns.countplot(x=crime_df['SHOOTING'])

#%%[markdown]
# 2)Pie Chart
crime_df['SHOOTING'].value_counts().plot.pie(autopct = '%1.1f%%')
#%%[markdown]
# 3) Distribution for the occurance of Shooting over the Years
sns.countplot(x = crime_df['YEAR'], hue = crime_df['SHOOTING'])
plt.title('Distribution of Shooting')

#%%[markdown]
# 4) Distribution of Shooting District wise
crime_df[crime_df['SHOOTING'] == 'Y']['DISTRICT'].value_counts().plot(kind = 'bar', color = '#8CD9A3')
plt.title('Distribution of Shooting occuring in the district')
#%%
crime_df[crime_df['SHOOTING'] == 'N']['DISTRICT'].value_counts().plot(kind = 'bar', color = '#FF5C5C')
plt.title('Distribution of Shooting not occuring in the district')
#%%[markdown]
# Observations for shooting variable

#%%[markdown]
# Distribution of different offence District wise
order = crime_df['OFFENSE_CODE_GROUP'].value_counts().head(6)
order = order.drop('Other').index
sns.countplot(data = crime_df, x='OFFENSE_CODE_GROUP',hue='DISTRICT', order = order, palette = 'plasma')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
plt.xticks(rotation=75)
plt.show()

#%%
#Line graph(District, Year, Offense code group)
grouped = crime_df.groupby(['YEAR', 'DISTRICT'])['OFFENSE_CODE_GROUP'].count().reset_index()
sns.lineplot(data = grouped.reset_index(), x='YEAR', y='OFFENSE_CODE_GROUP',hue='DISTRICT', palette = 'plasma')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
plt.title('Line Graph for each district showing the number of incidents distributed over the years')
plt.show()
#%%[markdown]
#Line Graph for each district showing the number of incidents distributed over the years
#plotly
grouped = crime_df.groupby(['YEAR', 'DISTRICT'])['OFFENSE_CODE_GROUP'].count().reset_index()
fig = px.line(grouped, x='YEAR', y='OFFENSE_CODE_GROUP', color='DISTRICT', labels={'OFFENSE_CODE_GROUP': 'Number of Incidents', 'YEAR': 'YEAR'})

# Update layout for legend
fig.update_layout(
    title={
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    }
)

# Update layout for legend
fig.update_layout(legend=dict(
    title="District",
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
))
# Show the plot
fig.show()
#%%
#Line graph(District, Month, Offense code group)
grouped = crime_df.groupby(['MONTH', 'DISTRICT'])['OFFENSE_CODE_GROUP'].count().reset_index()
sns.lineplot(data = grouped.reset_index(), x='MONTH', y='OFFENSE_CODE_GROUP',hue='DISTRICT', palette = 'plasma')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
plt.title('Line Graph for each district showing the number of incidents distributed over the months')
plt.show()

#%%[markdown]
#Line Graph for each district showing the number of incidents distributed over the months
#plotly
grouped = crime_df.groupby(['MONTH', 'DISTRICT'])['OFFENSE_CODE_GROUP'].count().reset_index()
fig = px.line(grouped, x='MONTH', y='OFFENSE_CODE_GROUP', color='DISTRICT', labels={'OFFENSE_CODE_GROUP': 'Number of Incidents', 'MONTH': 'Month'})

# Update layout for legend
fig.update_layout(
    title={
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    }
)

# Update layout for legend
fig.update_layout(legend=dict(
    title="District",
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
))

fig.show()
#%%
#Pie chart for all the Offence Code Group
labels = crime_df['OFFENSE_CODE_GROUP'].astype('category').cat.categories.tolist()
counts = crime_df['OFFENSE_CODE_GROUP'].value_counts()
sizes = [counts[var_cat] for var_cat in labels]
fig1, ax1 = plt.subplots(figsize = (22,12))
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140) 
ax1.axis('equal')
plt.show()

# %%[markdown]
# Bivariate Analysis

# %%
numerical_columns = crime_df.select_dtypes(include=['int64']).columns
len(numerical_columns)
#%%
# Determine the number of rows/columns for the subplot grid
n_cols = 3  
n_rows = (len(numerical_columns) + n_cols - 1) // n_cols

# Create a figure and a grid of subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*5))

# Flatten the axes array for easy iteration
axes = axes.flatten()

for i, col in enumerate(numerical_columns):
    sns.boxplot(x='SHOOTING', y=col, data=crime_df, ax=axes[i], color = '#d53e4f')
    axes[i].set_title(col)
    axes[i].set_xlabel('SHOOTING')
    axes[i].set_ylabel('')


plt.tight_layout()
plt.show()
#%%
#Chi-squared Test
categorical_columns = crime_df.select_dtypes(include=['object', 'category']).columns

for col in categorical_columns:
    # Create a cross-tabulation
    contingency_table = pd.crosstab(crime_df[col], crime_df['SHOOTING'])
    # Perform the Chi-squared test
    chi2, p, dof, ex = chi2_contingency(contingency_table)
    print(f"Chi-squared test for {col}: p-value = {p}")
#%%
# Selecting categorical columns
categorical_columns = crime_df.select_dtypes(include=['object', 'category']).columns
# len(categorical_columns)
#%%
# Create a figure and a grid of subplots
fig, axes = plt.subplots(7, 1, figsize=(25,20))

# Flatten the axes array for easy iteration
axes = axes.flatten()

for index,col in enumerate(categorical_columns):
    sns.countplot(x=col, hue='SHOOTING', data=crime_df, ax=axes[index], palette='Set2')

plt.tight_layout()
plt.show()
# %%[markdown]
# Observation for bivariate analysis

#%%
# Assuming Shooting is a Yes/No variable, we convert it to 1/0
crime_df['SHOOTING'] = crime_df['SHOOTING'].map({'Y': 1, 'N': 0})

# %%[markdown]
# Correlation Matrix
correlations = crime_df[['YEAR','MONTH','HOUR','DATE','SHOOTING','Lat','Long']].corr()
correlations
#%%
shooting_correlations = correlations['SHOOTING'].sort_values()
print(shooting_correlations)
#%%[markdown]
# Heatmap for the correlation matrix
sns.heatmap(correlations,annot = True)

#%%
# Scatterplot using plotly
correlations = crime_df[['YEAR','MONTH','HOUR','DATE','SHOOTING','Lat','Long']].corr()

# Create the heatmap
fig = px.imshow(correlations, text_auto=True, aspect="auto", title='Correlation Heatmap')
fig.show()
#%%[markdown]
# Multivariate Analysis
#%%[markdown]
# Scatterplot for Latitude and Longitude
custom_colors = [ '#1f77b4',  '#ff7f0e',   '#2ca02c',   '#d62728',   '#9467bd',   '#8c564b',  '#e377c2',   '#7f7f7f',   '#bcbd22',   '#17becf',   '#aec7e8',  '#ffbb78']
sns.scatterplot(data=crime_df, x='Long', y='Lat', hue='DISTRICT', alpha=0.5, palette = custom_colors)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
plt.title('Scatter-plot for Latitude and Longitude')
plt.xlim(-71.200,-71.000)
plt.ylim(42.200,42.400)

#%%
# For time series analysis 
#%%
crime_df = pd.read_csv("cleaned_data.csv", encoding='latin1')
crime_df.shape
crime_df.columns

# Run this cell before droping few columns
#%%[markdown]
#Time series analysis

crime_df['SHOOTING'] = crime_df['SHOOTING'].map({'Y': 1, 'N': 0})
crime_df['OCCURRED_ON_DATE'] = pd.to_datetime(crime_df['OCCURRED_ON_DATE'])
crime_df.set_index('OCCURRED_ON_DATE', inplace=True)

# Resample to a larger time frame (e.g., monthly) if needed
shooting_time_series = crime_df.resample('M')['SHOOTING'].sum()
shooting_time_series.plot()
plt.show()

#%%
#TSA using plotly

# Convert 'OCCURRED_ON_DATE' to datetime and sort
crime_df['OCCURRED_ON_DATE'] = pd.to_datetime(crime_df['OCCURRED_ON_DATE'])
crime_df.sort_values('OCCURRED_ON_DATE', inplace=True)

# Group by 'OCCURRED_ON_DATE' and 'SHOOTING', then count the incidents
grouped_df = crime_df.groupby([pd.Grouper(key='OCCURRED_ON_DATE', freq='M'), 'SHOOTING']).size().reset_index(name='NUM_INCIDENTS')

# Create the line plot
fig = px.line(grouped_df, x='OCCURRED_ON_DATE', y='NUM_INCIDENTS', color='SHOOTING', title='Monthly Trend of Incidents Over Time by Shooting')
fig.show()
# %%
