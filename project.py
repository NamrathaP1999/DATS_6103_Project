#%%
# Importing necessary libraries and packages
# This includes pandas for data manipulation, numpy for numerical operations,
# matplotlib and seaborn for data visualization, and other essential libraries
# required for data analysis and modeling tasks.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Additional libraries will be imported as and when needed based on the specific requirements of the analysis

#%%
# Loading the crime dataset into a pandas DataFrame
# The dataset is read from a CSV file named 'crime.csv'
# The 'encoding' parameter is set to 'latin1' to handle special characters in the data that may not be properly interpreted using the default UTF-8 encoding
crime_data = pd.read_csv("crime_data.csv", encoding='latin1')

#%% [markdown]
# I) Data Cleaning and Data Preprocessing

#%%
# Displaying the first few rows and total length of the crime dataset
# This helps in getting a quick overview of the data structure, column names,
# and some initial data points for a preliminary understanding of the dataset
print(crime_data.head())
print(len(crime_data))

# Checking the data types of each column in the crime dataset
# This is a crucial step for data sanity check to ensure that each column is of the appropriate data type

#%%
crime_data.info()
crime_data.describe()

#%%
# Addressing Missing Values and Inappropriate Data Types
# This section focuses on cleaning the dataset by handling missing (null) values in certain features.
# Appropriate methods such as imputation or removal will be employed to ensure data completeness.
# Additionally, the data types of some features are not optimal for analysis. These will be altered
# to more suitable types (e.g., converting to categorical or datetime types where necessary).
# User-defined functions will be utilized in this process to streamline the data cleaning and
# type conversion tasks, enhancing the dataset's readiness for subsequent analysis.

#%%
# 1) Analyzing `INCIDENT_NUMBER` column
# Verifying the Uniqueness of the 'INCIDENT_NUMBER' Column
# This step involves checking whether each entry in the 'INCIDENT_NUMBER' column is unique.
# If every incident number is found to be unique, it can serve as a primary key for the dataset,
# ensuring that each row represents a distinct incident and thereby maintaining data integrity.
number_of_unique_rows = crime_data['INCIDENT_NUMBER'].nunique()
print("\nTotal Number of Unique Values in the Incident Number column: ", number_of_unique_rows)

# Observing Duplicate Entries in 'INCIDENT_NUMBER'
# An initial examination reveals that only 282,517 out of 319,073 'INCIDENT_NUMBER' values are unique,
# indicating that nearly 18% of the entries might be duplicates.
# This raises important questions for the data analysis:
# - Are these duplicates indicative of multiple records for a single incident?
# - Could they represent complex incidents involving multiple offenses?
# It is crucial to investigate these duplicates further to understand their nature.

#%%
# Verifying the presence of duplicate rows i.e. to check whether single incident is reported multiple times
# Group by 'INCIDENT_NUMBER'
grouped = crime_data.groupby('INCIDENT_NUMBER')

# Initialize a counter for incidents with multiple distinct entries
multiple_entries_count = 0

# Initialize a counter for printing sets
print_counter = 0

# Iterate through each group
for name, group in grouped:
    # Drop the 'INCIDENT_NUMBER' column as it's the same for the group
    group_without_id = group.drop(columns=['INCIDENT_NUMBER'])

    # Check if there are distinct rows in the group
    if group_without_id.drop_duplicates().shape[0] > 1:
        # Increment counter if more than one distinct entry is found
        multiple_entries_count += 1

        # Print the first three sets of rows
        if print_counter < 3:
            print(f"Incident number: {name}")
            print(group)
            print("\n")  
            print_counter += 1

print("Number of incidents with multiple distinct entries:", multiple_entries_count)

# Analysis of Duplicate Incident Numbers with Distinct Offenses
# The investigation into the dataset reveals that there are no exact duplicate rows present.
# Despite having the same 'INCIDENT_NUMBER', different entries vary in terms of 'OFFENSE_CODE' and 'UCR_PART'.
# This observation suggests that a single incident can involve multiple offenses.
# Each entry with the same incident number but different offense codes or UCR parts
# represents a different aspect or development in the incident.
# This highlights the complexity of certain incidents and underscores the importance
# of considering the entire context of each incident number in the analysis.

#%%
# Set multi index to have unique index
#crime_data.set_index(['INCIDENT_NUMBER', 'OFFENSE_CODE'], inplace = False)

#%%
# 2) Analyzing 'OFFENSE_CODE' Column
# First, determine the number of unique offense codes present in the dataset
unique_code = crime_data['OFFENSE_CODE'].value_counts()
number_of_unique_code = crime_data['OFFENSE_CODE'].nunique()
print(unique_code)
print(number_of_unique_code)

#%% 
# Next, check for any missing values in 'OFFENSE_CODE'
num_missing_code = crime_data['OFFENSE_CODE'].isnull().sum()
print("\nNumber of missing values in OFFENSE_CODE: ", num_missing_code)

# The analysis shows there are no missing values in 'OFFENSE_CODE'
# However, the data type of this column is currently 'int64', which may not be optimal
# Converting 'OFFENSE_CODE' to a 'categorical' data type can be beneficial because
# categorical treatment allows for easy grouping and analysis of different crime types without implying any numerical relationship between them.

#%%
# Convert 'OFFENSE_CODE' column to a categorical data type
crime_data['OFFENSE_CODE'] = crime_data['OFFENSE_CODE'].astype('category')

# Verify the conversion
print(crime_data['OFFENSE_CODE'].dtype)

#%%
# 3) Analyzing `OFFENSE_CODE_GROUP` Column
# First, determine the number of unique offense code groups present in the dataset
unique_code_group = crime_data['OFFENSE_CODE_GROUP'].value_counts()
number_of_unique_code_group = crime_data['OFFENSE_CODE_GROUP'].nunique()
print(unique_code_group)
print(number_of_unique_code_group)

#%% 
# Next, check for any missing values in 'OFFENSE_CODE_GROUP'
num_missing_code_group = crime_data['OFFENSE_CODE_GROUP'].isnull().sum()
print("\nNumber of missing values in OFFENSE_CODE: ", num_missing_code_group)

#%%
# It is observed that the 'OFFENSE_CODE_GROUP' column contains no missing values,
# which is beneficial for a complete analysis. However, its current data type is 'object'.
# For improved clarity and consistency in data handling, it would be advantageous to convert this column to 'string' type.
# Convert 'OFFENSE_CODE_GROUP' column to string data type
crime_data['OFFENSE_CODE_GROUP'] = crime_data['OFFENSE_CODE_GROUP'].astype('string')

# Verify the conversion
print(crime_data['OFFENSE_CODE_GROUP'].dtype)

#%%
# 4) Analyzing `OFFENSE_DESCRIPTION` Column
# First, determine the number of unique offense description present in the dataset
unique_description = crime_data['OFFENSE_DESCRIPTION'].value_counts()
number_unique_description = crime_data['OFFENSE_DESCRIPTION'].nunique()
print(unique_description)
print(number_unique_description)

#%% 
# Next, check for any missing values in 'OFFENSE_DESCRIPTION'
num_missing_description = crime_data['OFFENSE_CODE_GROUP'].isnull().sum()
print("\nNumber of missing values in OFFENSE_CODE: ", num_missing_description)

#%%
# It is observed that the 'OFFENSE_DESCRIPTION' column contains no missing values,
# However, its current data type is 'object'.
# For improved clarity and consistency in data handling, it would be advantageous to convert this column to 'string' type.
# Convert 'OFFENSE_DESCRIPTION' column to string data type
crime_data['OFFENSE_DESCRIPTION'] = crime_data['OFFENSE_DESCRIPTION'].astype('string')

# Verify the conversion
print(crime_data['OFFENSE_DESCRIPTION'].dtype)

#%%
# 5) Analyzing `DISTRICT` Column
# First, determine the number of unique districts present in this column
unique_districts = crime_data['DISTRICT'].value_counts()
num_unique_dstricts = crime_data['DISTRICT'].nunique()
print(unique_districts)
print(num_unique_dstricts)

#%% 
# Next, check for any missing values in 'DISTRICT' column
num_missing_districts = crime_data['DISTRICT'].isnull().sum()
print("\nNumber of missing values in DISTRICT: ", num_missing_districts)

#%%
# To impute missing district values using the 'REPORTING_AREA' information, 
# a relationship between 'REPORTING_AREA' and 'DISTRICT' should be established. 
# The idea is to find the most common 'DISTRICT' for each 'REPORTING_AREA',
# and then use this information to fill in missing 'DISTRICT' values.
# Step 1: Find the Most Common District for Each Reporting Area
most_common_district = crime_data.groupby(['REPORTING_AREA', 'DISTRICT']).size().reset_index(name='count')
most_common_district = most_common_district.sort_values(['REPORTING_AREA', 'count'], ascending=False).drop_duplicates('REPORTING_AREA').set_index('REPORTING_AREA')['DISTRICT']

# Step 2: Impute Missing District Values
crime_data['DISTRICT'] = crime_data.apply(
    lambda row: most_common_district[row['REPORTING_AREA']] if pd.isnull(row['DISTRICT']) and row['REPORTING_AREA'] in most_common_district else row['DISTRICT'],
    axis=1
)

#%% 
# Verify missing values in 'DISTRICT' column
num_missing_districts = crime_data['DISTRICT'].isnull().sum()
print("\nNumber of missing values in DISTRICT: ", num_missing_districts)

#%%
# It is observed that the 'DISTRICT' column contains no missing values now,
# However, its current data type is 'object'.
# For improved clarity and consistency in data handling, it would be advantageous to convert this column to 'string' type.
# Convert 'DISTRICT' column to string data type
crime_data['DISTRICT'] = crime_data['DISTRICT'].astype('string')

# Verify the conversion
print(crime_data['DISTRICT'].dtype)

#%%
# Updating District Codes with Real-Time District Names
# The current dataset uses district codes, which are not immediately informative in real-world context.
# After referencing the Boston Police Department's website, it's identified that these codes correspond to specific district names.
# To enhance the dataset's readability and practical relevance, this step involves replacing the district codes
# with their actual names. This will make the dataset more intuitive and meaningful for analysis and interpretation.

# Define a function to replace a single district code with its name
def replace_district_code(district_code):
    # Mapping of district codes to names
    district_name_mapping = {
        'A1' : 'Downtown',
        'A15': 'Charlestown',
        'A7': 'East Boston',
        'B2': 'Roxbury',
        'B3': 'Mattapan',
        'C6': 'South Boston',
        'C11': 'Dorchester',
        'D4': 'South End',
        'D14': 'Brighton',
        'E5': 'West Roxbury',
        'E13': 'Jamaica Plain',
        'E18': 'Hyde Park'
    }
    
    # Return the corresponding district name or the original code if not found in the mapping
    return district_name_mapping.get(district_code, district_code)

# Apply the function to each row for the 'DISTRICT' column
crime_data['DISTRICT'] = crime_data['DISTRICT'].apply(replace_district_code)

#%%
# Verify the change
unique_districts = crime_data['DISTRICT'].value_counts()
num_unique_dstricts = crime_data['DISTRICT'].nunique()
print("\nUnique Districts count: ", unique_districts)
print("\nNumber of Unique Districts: ", num_unique_dstricts)

#%%
# 6) Analyzing `Reporting Area` column
# First, determine the number of unique reporting areas present in this column
unique_reporting_areas = crime_data['REPORTING_AREA'].value_counts()
num_unique_reporting_areas = crime_data['REPORTING_AREA'].nunique()
print(unique_reporting_areas)
print(num_unique_reporting_areas)

#%% 
# Next, check for any missing values in 'REPORTING_AREA' column
num_missing_reporting_area = crime_data['REPORTING_AREA'].isnull().sum()
print("\nNumber of missing values in REPORTING_AREA: ", num_missing_reporting_area)

# Regarding the 'REPORTING_AREA' column:
# As of now, there are no missing values in this column. 
# We do not plan to use 'REPORTING_AREA' in our initial analysis.
# If needed in later stages, we will process and include it accordingly.

#%%
# 7) Analyzing `SHOOTING` Column
# First, determine the number of unique shooting values present in this column
unique_shooting_values = crime_data['SHOOTING'].value_counts()
num_unique_shooting_values = crime_data['SHOOTING'].nunique()
print(unique_shooting_values)
print(num_unique_shooting_values)

# Converting 'SHOOTING' Column to Categorical Type
# The 'SHOOTING' column is currently an object type with 'Y' indicating a shooting incident.
# This column will be transformed such that all values are standardized to either 'Y' for shootings or 'N' for non-shootings.
# First, all values that are not 'Y' are replaced with 'N'. 
# Then, the column is converted to a categorical data type for efficient storage and processing.
# This conversion facilitates clearer and more consistent data analysis regarding shooting incidents.

#%%
# Replace all non-'Y' values in 'SHOOTING' with 'N'
crime_data['SHOOTING'] = crime_data['SHOOTING'].apply(lambda x: 'N' if x != 'Y' else 'Y')

# Convert 'SHOOTING' column to a categorical data type
crime_data['SHOOTING'] = crime_data['SHOOTING'].astype('category')

#%%
# Verify the conversion
print(crime_data['SHOOTING'].unique())
print(crime_data['SHOOTING'].dtype)
print(crime_data['SHOOTING'].value_counts())

#%% 
# Next, check for any missing values in 'SHOOTING' column
num_missing_shooting = crime_data['SHOOTING'].isnull().sum()
print("\nNumber of missing values in REPORTING_AREA: ", num_missing_shooting)

# Status of 'SHOOTING' Column
# It is observed that the `SHOOTING` column contains no missing values,
# and have only two categories, 'Y' and 'N'.

#%%
# 8) Analyzing `OCCURRED_ON_DATE` Column (date and time of occurrence of crime)
# First, determine the number of unique occurences present in this column
unique_shooting_values = crime_data['OCCURRED_ON_DATE'].value_counts()
num_unique_shooting_values = crime_data['OCCURRED_ON_DATE'].nunique()
print(unique_shooting_values)
print(num_unique_shooting_values)

#%% 
# Next, check for any missing values in 'REPORTING_AREA' column
num_missing_reporting_area = crime_data['REPORTING_AREA'].isnull().sum()
print("\nNumber of missing values in REPORTING_AREA: ", num_missing_reporting_area)

#%%
crime_data['OCCURRED_ON_DATE'].info()
crime_data['OCCURRED_ON_DATE'].describe()

# Status of 'OCCURRED_ON_DATE' Column
# Currently, the 'OCCURRED_ON_DATE' column has no missing values, and its data type is appropriate for analysis.
# If further analysis reveals the need for any modifications in this column, such changes will be implemented as required.

#%%
# 9) Analyzing `YEAR` Column
# First, determine the number of unique YEAR's present in this column
unique_year = crime_data['YEAR'].value_counts()
num_unique_years = crime_data['YEAR'].nunique()
print(unique_year)
print(num_unique_years)

#%% 
# Next, check for any missing values in 'YEAR' column
num_missing_year = crime_data['YEAR'].isnull().sum()
print("\nNumber of missing values in YEAR: ", num_missing_year)
print(crime_data['YEAR'].dtype)

# %%
# Convert 'YEAR' column to a categorical data type
crime_data['YEAR'] = crime_data['YEAR'].astype('category')

# Verify the conversion
print(crime_data['YEAR'].dtype)

# Status of the 'YEAR' Column
# The 'YEAR' column currently has no missing values and is stored as a categorical data type. 
# This format is chosen due to its utility in grouping, visualizing, and summarizing the data effectively.
# Should the need arise for numerical operations such as sorting or comparisons, 
# the column can be converted to an integer data type at that stage of the analysis.

# %%
# 10) Analyzing `MONTH` Column
# First, determine the number of unique MONTH's present in this column
unique_month = crime_data['MONTH'].value_counts()
num_unique_months = crime_data['MONTH'].nunique()
print(unique_month)
print(num_unique_months)

# %%
# Sanity check
crime_data['MONTH'].info()
crime_data['MONTH'].describe()

# %%
# Next, check for any missing values in 'MONTH' column
num_missing_month = crime_data['MONTH'].isnull().sum()
print("\nNumber of missing values in MONTH: ", num_missing_month)

# Status of the 'MONTH' Column
# The 'MONTH' column in the dataset is complete with no missing values.
# Additionally, the current data type of the 'MONTH' column is appropriate for the type of data it represents,
# which is beneficial for any analysis involving time-based trends or patterns.

# %%
# 11) Analyzing `DAY_OF_WEEK` Column
# First, determine the number of unique values present in this column
unique_day = crime_data['DAY_OF_WEEK'].value_counts()
num_unique_day = crime_data['DAY_OF_WEEK'].nunique()
print(unique_day)
print(num_unique_day)

# %%
# Sanity check
crime_data['DAY_OF_WEEK'].info()
crime_data['DAY_OF_WEEK'].describe()

#%%
# Next, check for any missing values in 'DAY_OF_WEEK' column
num_missing_day = crime_data['DAY_OF_WEEK'].isnull().sum()
print("\nNumber of missing values in DAY_OF_WEEK: ", num_missing_day)

# Converting 'DAY_OF_WEEK' Column to Categorical Type
# The 'DAY_OF_WEEK' column, representing days of the week.
# This change is advantageous because:
# It simplifies operations like sorting, grouping, and filtering, making them faster and more intuitive.
# It clearly defines the column as containing a limited and fixed set of values (the days of the week),
# which is semantically more appropriate for a categorical variable.
# This conversion is expected to facilitate more efficient and effective data analysis involving days of the week.

#%%
crime_data['DAY_OF_WEEK'] = crime_data['DAY_OF_WEEK'].astype('category')

# Verify the conversion
print(crime_data['DAY_OF_WEEK'].dtype)

# Status of the 'DAY_OF_WEEK' Column
# The 'DAY_OF_WEEK' column in the dataset is complete with no missing values.
# Additionally, the current data type of the 'DAY_OF_WEEK' column is appropriate for the type of data it represents,

# %%
# 12) Analyzing `HOUR` Column
# First, determine the number of unique values present in this column
unique_HOUR = crime_data['HOUR'].value_counts()
num_unique_HOUR = crime_data['HOUR'].nunique()
print(unique_HOUR)
print(num_unique_HOUR)

# %%
# Sanity check
crime_data['HOUR'].info()
crime_data['HOUR'].describe()

# %%
# Next, check for any missing values in 'HOUR' column
num_missing_HOUR = crime_data['HOUR'].isnull().sum()
print("\nNumber of missing values in HOUR: ", num_missing_HOUR)

# Status of the 'HOUR' Column
# The 'HOUR' column in the dataset is complete with no missing values.
# Additionally, the current data type of the 'HOUR' column is appropriate for the type of data it represents,
# which is beneficial for any analysis involving time-based trends or patterns.
# This makes it well-suited for analyses that focus on hourly variations in data, 
# such as studying crime trends throughout the day.

#%%
# 13) Analyzing 'UCR_PART' column
# First, determine the number of unique values present in this column
unique_UCR_PART = crime_data['UCR_PART'].value_counts()
num_unique_UCR_PART = crime_data['UCR_PART'].nunique()
print(unique_UCR_PART)
print(num_unique_UCR_PART)

#%%
# Sanity check
crime_data['UCR_PART'].info()
crime_data['UCR_PART'].describe()

# %%
# Next, check for any missing values in 'UCR_PART' column
num_missing_UCR_PART = crime_data['UCR_PART'].isnull().sum()
print("\nNumber of missing values in UCR_PART: ", num_missing_UCR_PART)

# There are 90 missing values in UCR_PART
# Impute Missing 'UCR_PART' Values Based on 'OFFENSE_CODE_GROUP'
# This code identifies the distribution of 'OFFENSE_CODE_GROUP' for records where 'UCR_PART' values are missing.
# By examining 'offense_code_for_missing_UCR_PART', we aim to determine the most common 'OFFENSE_CODE_GROUP'
# associated with missing 'UCR_PART' values. 
# This approach assumes a correlation between the offense code group and the UCR part classification, 
# enabling more accurate imputation of missing data.

# %%
offense_code_for_missing_UCR_PART = crime_data['OFFENSE_CODE_GROUP'][crime_data['UCR_PART'].isnull()]
print(offense_code_for_missing_UCR_PART.value_counts())

# Analyzing Specific Offense Code Groups Based on UCR_PART Classification for missing values of 'UCR_PART'
# There are four distinct offense code groups: 'HUMAN TRAFFICKING', 
# 'HUMAN TRAFFICKING - INVOLUNTARY SERVITUDE', 'HOME INVASION', and 'INVESTIGATE PERSON' for those missing values of 'UCR_PART'
# According to the classification provided by the Uniform Crime Reporting (UCR) Offence types, 
# defined by The Federal Bureau of Investigation (FBI) for reporting data on crimes,
# the groups 'HUMAN TRAFFICKING', 'HUMAN TRAFFICKING - INVOLUNTARY SERVITUDE', 
# and 'HOME INVASION' are categorized under 'PART ONE'. This category typically includes serious crimes.
# On the other hand, 'INVESTIGATE PERSON' falls under 'PART THREE', 
# which generally encompasses less severe offenses.
# This classification is crucial for understanding the severity and nature of the crimes represented in the dataset.

#%%
# User defined function for imputing
def impute_ucr_part_subset(data, subset):
    """
    Impute missing UCR_PART values in a subset of the DataFrame based on OFFENSE_CODE_GROUP.
    
    :param data: pandas DataFrame containing 'UCR_PART' and 'OFFENSE_CODE_GROUP' columns
    :param subset: Series containing the OFFENSE_CODE_GROUPs of rows with missing UCR_PART
    """
    # Mapping from OFFENSE_CODE_GROUP to UCR_PART
    ucr_part_mapping = {
        'HUMAN TRAFFICKING': 'Part One',
        'HUMAN TRAFFICKING - INVOLUNTARY SERVITUDE': 'Part One',
        'HOME INVASION': 'Part One',
        'INVESTIGATE PERSON': 'Part Three'
    }

    # Find the indices in the main DataFrame that correspond to the subset
    indices_to_impute = subset.index

    # Impute missing UCR_PART values in the subset
    for index in indices_to_impute:
        offense_code_group = data.at[index, 'OFFENSE_CODE_GROUP']
        if offense_code_group in ucr_part_mapping:
            data.at[index, 'UCR_PART'] = ucr_part_mapping[offense_code_group]

# Call the function
impute_ucr_part_subset(crime_data, offense_code_for_missing_UCR_PART)

# Check the results for the imputed subset
print(crime_data.loc[offense_code_for_missing_UCR_PART.index][['OFFENSE_CODE_GROUP', 'UCR_PART']])

#%% 
# Verify the change
unique_UCR_PART = crime_data['UCR_PART'].value_counts()
num_unique_UCR_PART = crime_data['UCR_PART'].nunique()
print(unique_UCR_PART)
print(num_unique_UCR_PART)

num_missing_UCR_PART = crime_data['UCR_PART'].isnull().sum()
print("\nNumber of missing values in UCR_PART: ", num_missing_UCR_PART)

# %%
# Converting 'UCR_PART' Column to Categorical Type
# because it simplifies operations like sorting, grouping, and filtering, based on UCR_PART, making them faster and more intuitive.
# Change the data type 
crime_data['UCR_PART'] = crime_data['UCR_PART'].astype('category')

#%%
# Verify the conversion
print(crime_data['UCR_PART'].dtype)

# Status of the 'UCR_PART' Column
# The 'UCR_PART' column in the dataset currently has no missing values.
# Additionally, the data type of this column is appropriate for the kind of information it represents.

# %%
# 14) Analyzing `STREET` Column
# First, determine the number of unique values present in this column
unique_street = crime_data['STREET'].value_counts()
num_unique_street = crime_data['STREET'].nunique()
print(unique_street)
print(num_unique_street)

# %%
# Sanity check
crime_data['STREET'].info()
crime_data['STREET'].describe()

#%% Convert the data type to 'STRING'
crime_data['STREET'] = crime_data['STREET'].astype('string')

#%%
# Verify the change
print(crime_data['STREET'].dtype)

# %%
# Next, check for any missing values in 'STREET' column
num_missing_STREET = crime_data['STREET'].isnull().sum()
print("\nNumber of missing values in STREET: ", num_missing_STREET)

# Handling Missing Values in 'STREET' Column
# The dataset contains 10,871 missing values in the 'STREET' column.
# To address these missing values without imputing specific street names,
# we categorize them as 'Not specified'. This allows us to maintain the integrity of the dataset 
# while acknowledging the absence of certain location details.

#%%
# Replace missing values in 'STREET' with 'Not specified'
crime_data['STREET'].fillna('Not Specified', inplace=True)

#%%
# Verify the operation
print(crime_data['STREET'].value_counts())
num_missing_STREET = crime_data['STREET'].isnull().sum()
print("\nNumber of missing values in STREET: ", num_missing_STREET)

# State of the 'STREET' Column
# There are no missing values present in the 'STREET' column of the dataset.
# Additionally, the data type of the 'STREET' column is appropriately set as a string, which is ideal for textual street name data.

#%%
# Export the dataframe to a CSV file
crime_data.to_csv('final_crime_data.csv', index = False)


# II) Exploratory Data Analysis 

# III) Predictive Analysis and Modeling 

#%%
# Final dataset
# Considering the interested columns.
crime_data = crime_data[['INCIDENT_NUMBER', 'OFFENSE_CODE', 'OFFENSE_CODE_GROUP', 'OFFENSE_DESCRIPTION', 'DISTRICT', 'REPORTING_AREA', 'SHOOTING', 'OCCURRED_ON_DATE', 'YEAR', 'MONTH', 'DAY_OF_WEEK', 'HOUR', 'UCR_PART', 'STREET']]

#%%
# In addressing the challenge posed by our imbalanced dataset, particularly for predicting shooting incidents, 
# we adopted a nuanced approach that focused on creating a more balanced dataset through stratified sampling based on the 'DISTRICT' variable. 
# The initial state of our dataset showed a significant imbalance, with a much higher number of non-shooting incidents (class 'N') compared to shooting incidents (class 'Y'). 
# This disparity presented a considerable challenge in accurately predicting the relatively rare, yet critical, shooting incidents.

# To counteract this, we employed a proportional subsetting strategy. While we retained all instances of reported shooting incidents (class 'Y'), 
# we carefully crafted a subset of non-reported shooting incidents (class 'N'). The key to this approach was to ensure that this subset was representative of the entire dataset in terms of the distribution across different districts.
# By focusing on 'DISTRICT', we aimed to preserve the underlying structure and distribution of the original data within our sample.

# This method of stratified sampling based on 'DISTRICT' allowed us to maintain a balanced representation of the dataset. 
# It was crucial to select 'DISTRICT' as the stratifying variable because of its potential impact on the occurrence and reporting of shooting incidents. 
# By doing so, our subset did not merely represent a random selection but was a deliberate representation of the geographical distribution of incidents.

# The impact of this method was evident in the improved performance of our predictive model. 
# There was a marked enhancement in the model's ability to predict shooting incidents, as reflected in the improved precision, recall, and F1-score for the minority class ('Y'). 
# The model's improved performance was further substantiated by a high ROC AUC score, indicating a robust ability to differentiate between the two classes.

# %%
# Group by 'DISTRICT' and then sample from each group
grouped = crime_data[crime_data['SHOOTING'] == 'N'].groupby('DISTRICT')
sampled_non_reported = grouped.apply(lambda x: x.sample(min(len(x), int(10000 * len(x) / len(crime_data)))))
sampled_non_reported = sampled_non_reported.reset_index(drop=True)

reported_shootings = crime_data[crime_data['SHOOTING'] == 'Y']
balanced_dataset = pd.concat([reported_shootings, sampled_non_reported], ignore_index=True)

balanced_dataset = balanced_dataset.sample(frac=1, random_state=42).reset_index(drop=True)
balanced_dataset.to_csv('Balanced_data.csv', index = False)

# After implementing the stratified sampling technique to balance our dataset, particularly focusing on the 'DISTRICT' variable, the resultant dataset was saved as a CSV file. 
# This step is crucial for ensuring consistency in our modeling process. The reason behind saving the stratified sample to a CSV file stems from the nature of our sampling method: 
# each execution of the stratified sampling code can potentially yield a slightly different dataset due to the randomness inherent in the sampling process.
# By saving the stratified dataset as a CSV, we establish a fixed dataset that can be reliably used for all subsequent modeling. 
# This approach eliminates the variability that would arise from repeatedly running the stratified sampling process, which could lead to different subsets of data and, consequently, different modeling outcomes.
# Using a fixed CSV file for modeling ensures that our results are reproducible and consistent, a key aspect in the validation of machine learning models.
# The commented code related to the stratification process serves as a record of the methodology used to obtain the balanced dataset. 
# However, for the actual modeling, the saved CSV file is used. This practice enhances the reliability of our model by providing a stable and consistent dataset for training and testing, 
# thereby allowing for a more accurate assessment of the model's performance.


# I) Logistic Regression to predict 'SHOOTING'

# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

# Load the balanced_dataset
crime_data = pd.read_csv('Balanced_data.csv')

# Selecting features and target variable
X = crime_data.drop(['OFFENSE_CODE', 'INCIDENT_NUMBER', 'OCCURRED_ON_DATE', 'SHOOTING'], axis=1)
y = crime_data['SHOOTING'].apply(lambda x: 1 if x == 'Y' else 0) 

# Handling categorical variables
categorical_features = ['OFFENSE_CODE_GROUP', 'DISTRICT', 'YEAR', 'DAY_OF_WEEK', 'UCR_PART', 'STREET']
numerical_features = ['MONTH', 'HOUR']

# Create a column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)  
    ])

# Create a pipeline that combines the preprocessor with a logistic regression model
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', LogisticRegression(max_iter=1000))])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit the model
pipeline.fit(X_train, y_train)

# Predictions
y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

#%%
# Evaluation
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred_proba))

#%%
# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % roc_auc_score(y_test, y_pred_proba))
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

# II) DECISION TREE (CLASSIFICATION TREE)

#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt

# Load the balanced dataset
crime_data = pd.read_csv('Balanced_data.csv')

# Selecting features and target variable
X = crime_data.drop(['INCIDENT_NUMBER', 'OCCURRED_ON_DATE', 'SHOOTING'], axis=1)
y = crime_data['SHOOTING'].apply(lambda x: 1 if x == 'Y' else 0)

# Handling categorical variables
categorical_features = ['OFFENSE_CODE', 'OFFENSE_CODE_GROUP', 'DISTRICT', 'YEAR', 'DAY_OF_WEEK', 'UCR_PART', 'STREET']
numerical_features = ['MONTH', 'HOUR']

# Create a column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)  
    ])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Apply preprocessing to the training and testing data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Create and fit the decision tree classifier
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train_processed, y_train)

# Predictions
y_pred_dt = dt_classifier.predict(X_test_processed)
y_pred_proba_dt = dt_classifier.predict_proba(X_test_processed)[:, 1]

#%%
# Evaluation
print("Decision Tree Classification Report:\n", classification_report(y_test, y_pred_dt))
print("Decision Tree Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))
print("Decision Tree ROC AUC Score:", roc_auc_score(y_test, y_pred_proba_dt))

#%%
# ROC Curve
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_pred_proba_dt)
plt.figure()
plt.plot(fpr_dt, tpr_dt, label='Decision Tree (area = %0.2f)' % roc_auc_score(y_test, y_pred_proba_dt))
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# III) KNN model

# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# Load the balanced_dataset
crime_data = pd.read_csv('Balanced_data.csv')

# Selecting features and target variable
X = crime_data.drop(['OFFENSE_CODE', 'INCIDENT_NUMBER', 'OCCURRED_ON_DATE', 'SHOOTING'], axis=1)
y = crime_data['SHOOTING'].apply(lambda x: 1 if x == 'Y' else 0)

# Handling categorical variables
categorical_features = ['OFFENSE_CODE_GROUP', 'DISTRICT', 'YEAR', 'DAY_OF_WEEK', 'UCR_PART', 'STREET']
numerical_features = ['MONTH', 'HOUR']

# Create a column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Create a pipeline that combines the preprocessor with a KNN classifier
# Note: Do not set n_neighbors here since we will determine the best k later
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', KNeighborsClassifier())])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# %%
# List to hold the average CV scores
cv_scores = []

# List to hold the values of k
k_values = list(range(1, 31))  # Testing k from 1 to 30

# Perform 10-fold cross-validation with each value of k
for k in k_values:
    # Update the classifier's n_neighbors parameter
    pipeline.set_params(classifier__n_neighbors=k)
    scores = cross_val_score(pipeline, X_train, y_train, cv=10, scoring='f1')
    cv_scores.append(scores.mean())

# %%
# Plot F1 score vs k
plt.figure(figsize=(12, 6))
plt.plot(k_values, cv_scores, marker='o')
plt.xlabel('Value of k for KNN')
plt.ylabel('Cross-Validated F1 Score')
plt.title('KNN Cross-Validated F1 Score for Different k Values')
plt.show()

# %%
# Select the best k
best_k = k_values[cv_scores.index(max(cv_scores))]
print(f'The best value of k is {best_k}')

# %%
# Update the pipeline with the best k found
pipeline.set_params(classifier__n_neighbors=best_k)

# Fit the model with the training set
pipeline.fit(X_train, y_train)

# Predictions
y_pred_knn = pipeline.predict(X_test)
y_pred_proba_knn = pipeline.predict_proba(X_test)[:, 1]

# %%
# Evaluation
print("KNN Classification Report:\n", classification_report(y_test, y_pred_knn))
print("KNN Confusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))
print("KNN ROC AUC Score:", roc_auc_score(y_test, y_pred_proba_knn))

# %%
# ROC Curve for KNN
fpr_knn, tpr_knn, thresholds_knn = roc_curve(y_test, y_pred_proba_knn)
plt.figure()
plt.plot(fpr_knn, tpr_knn, label='KNN (area = %0.2f)' % roc_auc_score(y_test, y_pred_proba_knn))
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for KNN')
plt.legend(loc="lower right")
plt.show()


# IV) K means clustering
# %%
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load data
crime_data = pd.read_csv('Balanced_data.csv')

# Select features for clustering
features = ['OFFENSE_CODE', 'MONTH', 'HOUR']  # Example features
X = crime_data[features]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans clustering
k = 3  # Number of clusters
kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

# Apply PCA and reduce to two dimensions for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

#%%
# Plotting the clusters
plt.figure(figsize=(8, 6))
for i in range(k):
    plt.scatter(X_pca[y_kmeans == i, 0], X_pca[y_kmeans == i, 1], 
                label = f'Cluster {i}')
plt.title('Clusters of Crimes')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()

# %%
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load data
crime_data = pd.read_csv('final_crime_data.csv')

# Encode the categorical data
le_offense = LabelEncoder()
le_district = LabelEncoder()

crime_data['OFFENSE_CODE_GROUP'] = le_offense.fit_transform(crime_data['OFFENSE_CODE_GROUP'])
crime_data['DISTRICT'] = le_district.fit_transform(crime_data['DISTRICT'])

# Now select your features for clustering
X = crime_data[['OFFENSE_CODE_GROUP', 'DISTRICT']]

# Perform KMeans clustering (now X is all numeric)
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

#%%
# Plot the WCSS to find the elbow
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# %%
# Choose the number of clusters you decided upon (say 3 from the elbow method)
k = 3 
kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
y_kmeans = kmeans.fit_predict(X)

#%%
# Plotting the clusters
plt.figure(figsize=(8, 6))
for i in range(k):
    plt.scatter(X[y_kmeans == i]['OFFENSE_CODE_GROUP'], X[y_kmeans == i]['DISTRICT'], label=f'Cluster {i}')
plt.title('Clusters of Crimes by Offense Code Group and District')
plt.xlabel('Encoded Offense Code Group')
plt.ylabel('Encoded District')
plt.legend()
plt.show()

# %%
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load data
crime_data = pd.read_csv('final_crime_data.csv')

# Map 'DAY_OF_WEEK' to numerical values
days = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7}
crime_data['DAY_OF_WEEK'] = crime_data['DAY_OF_WEEK'].map(days)

# Extract the relevant features for clustering
X = crime_data[['DAY_OF_WEEK', 'HOUR']]

# Scale 'HOUR' feature
scaler = StandardScaler()
X['HOUR'] = scaler.fit_transform(X[['HOUR']])

# Determine the number of clusters using the Elbow Method
wcss = []
for i in range(1, 11):  # Let's test for 1 to 10 clusters
    kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot the WCSS to find the elbow
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#%%
# Choose the number of clusters (k) based on the Elbow Method
k = 4  # for example
kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Plotting the clusters
plt.figure(figsize=(8, 6))
plt.scatter(X['HOUR'][y_kmeans == 0], X['DAY_OF_WEEK'][y_kmeans == 0], s=50, c='red', label='Cluster 1')
plt.scatter(X['HOUR'][y_kmeans == 1], X['DAY_OF_WEEK'][y_kmeans == 1], s=50, c='blue', label='Cluster 2')
plt.scatter(X['HOUR'][y_kmeans == 2], X['DAY_OF_WEEK'][y_kmeans == 2], s=50, c='green', label='Cluster 3')
plt.title('Clusters of Crimes by Day of Week and Hour')
plt.xlabel('Scaled Hour of Day')
plt.ylabel('Day of Week')
plt.xticks(ticks=range(1, 8), labels=list(days.keys()))  # Set x-ticks to day names
plt.legend()
plt.show()


# %%
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Assuming crime_data is already read and DAY_OF_WEEK is encoded as numeric
X = crime_data[['DAY_OF_WEEK', 'HOUR']]

# Apply KMeans
kmeans = KMeans(n_clusters=3, random_state=42)  # Adjust the number of clusters as needed
clusters = kmeans.fit_predict(X)

# Plotting the clusters
plt.figure(figsize=(10, 8))
plt.scatter(X['HOUR'], X['DAY_OF_WEEK'], c=clusters, cmap='viridis', label='Clusters')
plt.colorbar(ticks=[0, 1, 2], label='Cluster Label')  # Adjust number of ticks based on the number of clusters

# Annotating the plot with day names and hour
day_ticks = range(1, 8)  # 1 to 7 for Monday to Sunday
hour_ticks = range(0, 24)  # 0 to 23 for each hour
plt.xticks(hour_ticks, hour_ticks)  # Set x-ticks to hours
plt.yticks(day_ticks, ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])  # Set y-ticks to day names

plt.title('Clusters of Crimes by Day of Week and Hour')
plt.xlabel('Hour of Day')
plt.ylabel('Day of Week')
plt.show()

# %%
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt

# Load data
crime_data = pd.read_csv('final_crime_data.csv')

# Encode 'UCR_PART'
ucr_encoder = LabelEncoder()
crime_data['UCR_PART'] = ucr_encoder.fit_transform(crime_data['UCR_PART'])

# Extract the relevant features for clustering
X = crime_data[['UCR_PART', 'MONTH']]

# Optional: Scale 'MONTH' feature if necessary
# scaler = MinMaxScaler(feature_range=(1, 12))
# X['MONTH'] = scaler.fit_transform(X[['MONTH']])

# Determine the number of clusters using the Elbow Method
wcss = []
for i in range(1, 11):  # Let's test for 1 to 10 clusters
    kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot the WCSS to find the elbow
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#%%
# Choose the number of clusters (k) based on the Elbow Method
k = 4  # for example
kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Plotting the clusters
plt.figure(figsize=(8, 6))
# Here, we use the original 'MONTH' and 'UCR_PART' for plotting to keep the original data scale
plt.scatter(crime_data['MONTH'][y_kmeans == 0], crime_data['UCR_PART'][y_kmeans == 0], s=50, c='red', label='Cluster 1')
plt.scatter(crime_data['MONTH'][y_kmeans == 1], crime_data['UCR_PART'][y_kmeans == 1], s=50, c='blue', label='Cluster 2')
plt.scatter(crime_data['MONTH'][y_kmeans == 2], crime_data['UCR_PART'][y_kmeans == 2], s=50, c='green', label='Cluster 3')
plt.scatter(crime_data['MONTH'][y_kmeans == 3], crime_data['UCR_PART'][y_kmeans == 3], s=50, c='purple', label='Cluster 4')
plt.title('Clusters of Crimes by UCR Part and Month')
plt.xlabel('Month')
plt.ylabel('UCR Part')
plt.xticks(ticks=range(1, 13), labels=range(1, 13))  # Set x-ticks to months
plt.yticks(ticks=range(len(ucr_encoder.classes_)), labels=ucr_encoder.classes_)  # Set y-ticks to UCR parts
plt.legend()
plt.show()

# %%
