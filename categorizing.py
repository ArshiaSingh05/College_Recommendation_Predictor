import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import MinMaxScaler
# Categorizing the data
data=pd.read_csv('cleaned_data.csv')
# Ensure UG fee column is numeric
data['UG fee (tuition fee)'] = pd.to_numeric(data['UG fee (tuition fee)'], errors='coerce').fillna(0)
# Define budget category based on quartiles
def categorize_budget(fee):
    if fee == 0:
        return 'Not Provided'
    elif fee <= 60177.5:
        return 'Low'
    elif fee <= 970500:
        return 'Medium'
    else:
        return 'High'
# Apply the function
data['Budget Category'] = data['UG fee (tuition fee)'].apply(categorize_budget)
# Display result
print(data['Budget Category'].value_counts())

# Average Rating
rating_cols = ['Overall Rating', 'Academic (Rating)', 'Accommodation (Rating)', 
                'Faculty (Rating)', 'Infrastructure (Rating)', 'Placement (Rating)', 'Social Life (Rating)']

data['Average Rating'] = data[rating_cols].mean(axis=1)

# Create Placement vs Fee Ratio
data['Placement vs Fee Ratio'] = data['Placement (Rating)'] / (data['UG fee (tuition fee)'] + 1)

# Normalizing UG and PG fees

data.to_csv('cleaned_data.csv', index=False)
print("Successfully saved")
