import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle


class gd:
    def __init__(self, LR, Epoch, n_features):
        self.coef = None
        self.inter = None
        self.lr = LR
        self.epoch = Epoch
        self.n_features =n_features
        self.scaler = None
        self.location_mapping = None
        self.area_type_mapping = None

    def fit(self, x_train, y_train):
        self.inter = 0.0
        self.coef = np.ones((self.n_features, 1), dtype=float)

        for i in range(self.epoch):
            y_hat = np.dot(x_train, self.coef) + self.inter
            residuals = y_train - y_hat
            coef_der = -2 * np.dot(x_train.T, residuals) / x_train.shape[0]
            inter_der = -2 * np.mean(residuals)
            self.inter -= self.lr * inter_der
            self.coef -= self.lr * coef_der

    def predict_s(self, x_test):
        return np.dot(x_test, self.coef) + self.inter

    def set_mappings(self, location_mapping, area_type_mapping):
        self.location_mapping = location_mapping
        self.area_type_mapping = area_type_mapping

    def set_scaler(self, scaler):
        self.scaler = scaler

    def convert_input_data(self, input_data):
        # Convert categorical data to numerical using mappings
        input_data['site_location'] = input_data['location'].map(self.location_mapping).fillna(0)
        input_data['area_type'] = input_data['area_type'].map(self.area_type_mapping).fillna(0)

        # Drop the original categorical columns
        input_data = input_data.drop(columns=['location'])
        #
        #area_type,size,total_sqft,bath,balcony,price,site_location
        input_data = input_data.rename(columns={'total_sqft': 'total_sqft', 'bath': 'bath', 
                                            'balcony': 'balcony', 'site_location': 'site_location'})

        # Reorder the columns to match the training data order
        input_data = input_data[['area_type', 'size', 'total_sqft', 'bath', 'balcony', 'site_location']]

        return input_data


    def predict(self, input_data):
        # Convert the input data
        input_data_converted = self.convert_input_data(input_data)
    
        # Scale the input data using the fitted scaler
        input_data_scaled = self.scaler.transform(input_data_converted)
    
        # Predict using the gradient descent model
        predicted_price = self.predict_s(input_data_scaled)
        ans=round(predicted_price[0][0], 2)
    
        return ans


def convertRang(x):
    if isinstance(x, str):
        temp = x.split('-')
        if len(temp) == 2:
            return (float(temp[0]) + float(temp[1])) / 2
        try:
            return float(temp[0])
        except ValueError:
            return None
    return x

def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, sub_df in df.groupby('site_location'):
        m = np.mean(sub_df.price_per_sqft)
        sd = np.std(sub_df.price_per_sqft)
        reduce_df = sub_df[(sub_df.price_per_sqft > (m - sd)) & (sub_df.price_per_sqft < (m + sd))]
        df_out = pd.concat([df_out, reduce_df], ignore_index=True)
    return df_out


# Load the dataset
data = pd.read_csv("Pune_House_Data.csv")

# Fill missing values
data['balcony'].fillna(data['balcony'].mode()[0], inplace=True)
data['bath'].fillna(data['bath'].mode()[0], inplace=True)
data['area_type'].fillna(data['area_type'].mode()[0], inplace=True)
data['size'].fillna(data['size'].mode()[0], inplace=True)
data['total_sqft'].fillna(data['total_sqft'].mode()[0], inplace=True)
data['site_location'].fillna(data['site_location'].mode()[0], inplace=True)

# Dropping the rows with null values 
data = data.dropna()

# Extracting number of BHK from 'size' column
data['size'] = data['size'].str.extract('(\d+)').astype(float)

# Convert 'total_sqft' to a single value if it's a range
data['total_sqft'] = data['total_sqft'].apply(convertRang)

# Removing the rows in new_total_sqft column that have None values
data = data.dropna()

# Label locations with less than or equal to 10 occurrences as 'other'
locations = data.groupby('site_location')['site_location'].agg('count').sort_values(ascending=False)
Less_locations = locations[locations <= 10]
data['site_location'] = data['site_location'].apply(lambda x: 'other' if x in Less_locations else x)

unique_locations = data['site_location'].unique()
unique_area_types = data['area_type'].unique()

location_mapping = {location: index for index, location in enumerate(unique_locations)}
area_type_mapping = {area_type: index for index, area_type in enumerate(unique_area_types)}

data['site_location'] = data['site_location'].map(location_mapping)
data['area_type'] = data['area_type'].map(area_type_mapping)

# Add price_per_sqft column
data['price_per_sqft'] = data['price'] / data['total_sqft']

data = remove_pps_outliers(data)
# Drop price_per_sqft column
data = data.drop('price_per_sqft', axis=1)

# Separate features and target variable
x = data.drop('price', axis=1)
y = data['price'].values.reshape(-1, 1)

# Normalize the input features
scaler = StandardScaler()
x = scaler.fit_transform(x)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=20)

# Instantiate the gd class with a reduced learning rate
lr_model = gd(LR=0.01, Epoch=20000, n_features=X_train.shape[1])

# Train the model
lr_model.fit(X_train, y_train)

# Set mappings and scaler in the model
lr_model.set_mappings(location_mapping, area_type_mapping)
lr_model.set_scaler(scaler)

# Save the model to a pickle file
with open('lr_model.pkl', 'wb') as file:
    pickle.dump(lr_model, file)
