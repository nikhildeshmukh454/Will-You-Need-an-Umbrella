import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

# Read the dataset
data = pd.read_csv("C:\\Users\\nikhil deshmukh\\Desktop\\RESUME_PROJECT\\Rain\\weatherAUS.csv")

# Drop columns with a large number of missing values and other unnecessary columns
columns_to_drop = ["Evaporation", "Sunshine", "Cloud9am", "Cloud3pm", "Location", "Date", 
                   "WindGustDir", "WindDir9am", "WindDir3pm"]
data = data.drop(columns_to_drop, axis=1)

# Strip whitespace from column names
data.columns = data.columns.str.strip()

# Drop rows with any missing values
data = data.dropna()

# Initialize LabelEncoder
lb = LabelEncoder()

# Columns to encode
columns_to_encode = ['RainToday', 'RainTomorrow']

# Apply LabelEncoder to each column if it exists in the DataFrame
for column in columns_to_encode:
    if column in data.columns:
        data[column] = lb.fit_transform(data[column])

# Print counts before balancing
print("Counts before balancing:")
print(data.RainTomorrow.value_counts())

# Separate majority and minority classes
data_majority = data[data.RainTomorrow == 0]
data_minority = data[data.RainTomorrow == 1]

l=int(len(data_minority)*1.5)
# Downsample majority class
data_majority_downsampled = resample(data_majority, 
                                     replace=False,    # sample without replacement
                                     n_samples=l,  # to match minority class
                                     random_state=123)  # reproducible results

# Combine minority class with downsampled majority class
data_balanced = pd.concat([data_majority_downsampled, data_minority])

# Print counts after balancing
print("\nCounts after balancing:")
print(data_balanced.RainTomorrow.value_counts())

# Display the first 10 rows of the balanced DataFrame
print("\nFirst 10 rows of the balanced DataFrame:")
print(data_balanced.head(10))

# Save the balanced DataFrame to CSV
data_balanced.to_csv('modified_data.csv', index=False)

