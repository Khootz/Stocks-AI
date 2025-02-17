import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import math

# Load your dataset
file_path = 'C:\\Users\\User\\Desktop\\Stocks_AI\\combined_stock_data.csv'
df = pd.read_csv(file_path)

# Check for any NaNs in the dataset
df.dropna(inplace=True)  # Dropping NaNs, alternatively you can fill them

# Assuming you're predicting 'Close' price
# Feature columns can be modified as needed
feature_cols = ['Open', 'High', 'Low', 'Volume']
target_col = 'Close'

# Normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
df[feature_cols + [target_col]] = scaler.fit_transform(df[feature_cols + [target_col]])

# Split features and target
scaled_features = df[feature_cols].values
scaled_target = df[[target_col]].values

# Function to create a dataset for LSTM
def create_dataset(X, y, time_step=1):
    Xs, ys = [], []
    for i in range(len(X) - time_step):
        v = X[i:(i + time_step)]
        Xs.append(v)
        ys.append(y[i + time_step])
    return np.array(Xs), np.array(ys)

# Define time steps and reshape features for LSTM Model
time_step = 10
X, y = create_dataset(scaled_features, scaled_target, time_step)
y = y.reshape(-1, 1)  # Reshaping target to fit LSTM output

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, len(feature_cols))))
model.add(Dropout(0.1))  # Adding dropout
model.add(LSTM(50))
model.add(Dropout(0.1))  # Adding dropout
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Set up early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1, validation_split=0.1, callbacks=[early_stopping])

# Predict and measure RMSE
preds = model.predict(X_test)
preds = scaler.inverse_transform(preds)  # Inverse transform predictions
y_test_inv = scaler.inverse_transform(y_test)  # Inverse transform actuals

rmse = math.sqrt(mean_squared_error(y_test_inv, preds))
print("Test RMSE: ", rmse)
