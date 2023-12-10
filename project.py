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
# However, the data type of this column is currently 'object', which may not be optimal
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

# It is observed that the 'DISTRICT' column contains no missing values now,
# However, its current data type is 'object'.
# For improved clarity and consistency in data handling, it would be advantageous to convert this column to 'string' type.
# Convert 'DISTRICT' column to string data type
#%%
crime_data['DISTRICT'] = crime_data['DISTRICT'].astype('string')

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

# Optionally, to verify the conversion, you can check the data type again
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

# %%
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
# The 'MONTH' column in the dataset is complete with no missing values.
# Additionally, the current data type of the 'MONTH' column is appropriate for the type of data it represents,

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
#%%
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


# II) Exploratory Data Analysis 
