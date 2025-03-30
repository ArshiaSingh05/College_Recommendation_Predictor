import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv("College_data.csv")

#checking the data
print(data.head(),"\n")
print(data.info(),"\n")
print(data.isnull().sum(),"\n")
#only changing fees to int first
cols_to_check=data.columns[data.columns.get_loc('UG fee (tuition fee)'):]
data[cols_to_check]=data[cols_to_check].replace('--',pd.NA)
data_cleaned=data.dropna(subset=cols_to_check,how='all')
data_cleaned.reset_index(drop=True,inplace=True)
print(f"Original dataset size:{data.shape[0]} rows")
print(f"Cleaned dataset size:{data_cleaned.shape[0]} rows")

# Display rows with non-numeric values
print(data[['UG fee (tuition fee)', 'PG fee']][
    ~data['UG fee (tuition fee)'].astype(str).map(str.isdigit) | ~data['PG fee'].astype(str).map(str.isdigit)
])
print(data[['UG fee (tuition fee)', 'PG fee']].head())
print(data[['UG fee (tuition fee)', 'PG fee']].dtypes)

#filling other ratings with median of respective states
cols_to_fill=['Overall Rating','Academic (Rating)','Accommodation (Rating)','Faculty (Rating)','Infrastructure (Rating)','Placement (Rating)','Social Life (Rating)']
data[cols_to_fill] = data[cols_to_fill].apply(pd.to_numeric, errors='coerce')
for col in cols_to_fill:
    data[col].fillna(data.groupby('State')[col].transform('median'))

data[cols_to_fill]=data[cols_to_fill].fillna(data[cols_to_fill].median())

# Uniforming states with first letter as capital
data['State'] = data['State'].str.title().str.strip()

#filling the fee as same as ratings
cols_to_fill=['UG fee (tuition fee)','PG fee']
data[['UG fee (tuition fee)', 'PG fee']] = data[['UG fee (tuition fee)', 'PG fee']].apply(pd.to_numeric, errors='coerce')
for col in ['UG fee (tuition fee)', 'PG fee']:
    state_medians = data.groupby('State')[col].median()
    data[col] = data.apply(lambda row: state_medians[row['State']] if row[col] == 0 or pd.isna(row[col]) else row[col], axis=1)

#Checking outliers with boxplot

plt.figure(figsize=(12, 6))
sns.boxplot(data[['Overall Rating', 'Academic (Rating)', 'Accommodation (Rating)',
                  'Faculty (Rating)', 'Infrastructure (Rating)', 'Placement (Rating)', 
                  'Social Life (Rating)']])
plt.title("Boxplot of Rating Columns")
plt.show()
# Final checking
print("The final checks::")
print(data.describe().round(2))
print(data.info())
# Saving the changes
data.to_excel("cleaned_data.xlsx",index=False)
print("Successfully saved")
