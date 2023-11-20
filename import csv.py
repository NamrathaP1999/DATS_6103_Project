#%%

import pandas as pd

# Read the CSV file using the 'latin1' encoding
data = pd.read_csv("crime.csv", encoding='latin1')


# %%

crime = pd.DataFrame(data)
print(crime.head())
print(crime.dtypes)

#%%
# Assuming 'DISTRICT' column contains district codes
district_counts = crime['DISTRICT'].value_counts()

# Display the count of values for each district code
print(district_counts)

#%%

# Assuming 'DISTRICT' column contains district codes
district_counts = crime['DISTRICT'].value_counts()

# Create a dictionary to map district codes to groups
district_groups = {
    'A': ['A1', 'A15'],
    'B': ['B2', 'B3'],
    'C': ['C6', 'C11'],
    'D': ['D4', 'D14'],
    'E': ['E5', 'E13', 'E18']
}

# Initialize a dictionary to store counts for each group
group_counts = {group: 0 for group in district_groups}

# Sum the counts for each district code in its corresponding group
for group, codes in district_groups.items():
    group_counts[group] = district_counts[codes].sum()

# Display the counts for each group
print("Counts for Each Group:")
for group, count in group_counts.items():
    print(f"{group}: {count}")



# %%
# Create a new column 'ADDRESS' by combining relevant address-related columns
crime['ADDRESS'] = crime['STREET'].astype(str) + ', ' + crime['DISTRICT'].astype(str)

# Drop the individual address-related columns if needed
crime.drop(['STREET', 'DISTRICT'], axis=1, inplace=True)

# Display the updated DataFrame with the new 'ADDRESS' column
print(crime.head())

# %%
# Assuming 'DISTRICT' column contains district codes
district_counts = crime['ADDRESS'].value_counts()

# Display the count of values for each district code
print(district_counts)


# %%
