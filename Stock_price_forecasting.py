import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Download the stock data from Yahoo Finance
ticker_symbol = 'GOOG'
start_date = '2023-07-01'
end_date = '2024-07-01'

stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)

# Limit the dataset to the first 5000 data points
stock_data = stock_data.head(5000)

# Step 2: Handle missing values and calculate features
stock_data.ffill(inplace=True)

# Adding multiple features
stock_data['MA10'] = stock_data['Close'].rolling(window=10).mean()
stock_data['MA20'] = stock_data['Close'].rolling(window=20).mean()
stock_data['EMA10'] = stock_data['Close'].ewm(span=10, adjust=False).mean()
stock_data['EMA20'] = stock_data['Close'].ewm(span=20, adjust=False).mean()

# Calculate RSI
delta = stock_data['Close'].diff(1)
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
average_gain = gain.rolling(window=14).mean()
average_loss = loss.rolling(window=14).mean()
rs = average_gain / average_loss
stock_data['RSI'] = 100 - (100 / (1 + rs))

stock_data['Volume_Mean'] = stock_data['Volume'].rolling(window=10).mean()
stock_data['Daily Return'] = stock_data['Close'].pct_change()
stock_data['Lagged_Return_1'] = stock_data['Close'].shift(1).pct_change()
stock_data['Lagged_Return_2'] = stock_data['Close'].shift(2).pct_change()

# Forward fill to handle any missing values
stock_data.ffill(inplace=True)

# Ensure there are no remaining NaN or Inf values
stock_data.replace([np.inf, -np.inf], np.nan, inplace=True)
stock_data.dropna(inplace=True)

# Prepare the data for the model
features = ['MA10', 'MA20', 'EMA10', 'EMA20', 'RSI', 'Volume_Mean', 'Daily Return', 'Lagged_Return_1', 'Lagged_Return_2']
X = stock_data[features].values
y = stock_data['Close'].values
X = np.c_[np.ones(X.shape[0]), X]  # Add a column of ones for bias

# Standardize features manually
def standardize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1  # Avoid division by zero
    return (X - mean) / std

# Apply standardization (excluding the bias term)
X[:, 1:] = standardize(X[:, 1:])

# Ensure y is a column vector
y = y.reshape(-1, 1)

# Step 3: Define Huber Loss function
def huber_loss(y_true, y_pred, delta=20.0):
    error = y_true - y_pred
    is_small_error = np.abs(error) <= delta
    squared_loss = 0.5 * np.clip(error**2, -1e10, 1e10)  # Clip values to avoid overflow
    linear_loss = delta * (np.abs(error) - 0.5 * delta)
    return np.where(is_small_error, squared_loss, linear_loss).mean()

# Step 4: Implement Momentum Optimization
def momentum_optimization(X, y, learning_rate=0.0001, beta=0.9, iterations=45000, delta=1.2):
    m = X.shape[0]
    n = X.shape[1]
    theta = np.zeros((n, 1))  # Initialize theta as a column vector (n x 1)
    velocity = np.zeros((n, 1))  # Initialize velocity as a column vector (n x 1)
    
    loss_history = []  # List to store loss values for plotting
    
    for i in range(iterations):
        y_pred = X.dot(theta)  # (m x n) dot (n x 1) -> (m x 1)
        loss = huber_loss(y, y_pred, delta)
        loss_history.append(loss)  # Save the loss at each iteration
        
        # Check for NaN or Inf
        if np.isnan(loss) or np.isinf(loss):
            print(f"NaN or Inf encountered at iteration {i}, stopping optimization.")
            break
        
        gradient = -X.T.dot(y - y_pred) / m  # (n x m) dot (m x 1) -> (n x 1)
        velocity = beta * velocity + (1 - beta) * gradient
        theta -= learning_rate * velocity  # Update theta
        
        if i % 100 == 0:
            print(f"Iteration {i}: Huber Loss = {loss}")
    
    return theta, loss_history

# Train the model
theta, loss_history = momentum_optimization(X, y)

# Make predictions
y_pred = X.dot(theta)

# Flatten y_pred and y for visualization
y_pred = y_pred.flatten()
y = y.flatten()

# Step 5: Visualization

# Plot Actual vs Predicted Closing Prices
plt.figure(figsize=(10, 6))
plt.plot(stock_data.index[:len(y)], y, label='Actual Closing Prices', color='blue')
plt.plot(stock_data.index[:len(y_pred)], y_pred, label='Predicted Closing Prices', color='red')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title('Actual vs Predicted Closing Prices')
plt.legend()
plt.show()

# Plot Loss over Iterations (Epoch Plot)
plt.figure(figsize=(10, 6))
plt.plot(range(len(loss_history)), loss_history, label='Huber Loss', color='green')
plt.xlabel('Iteration')
plt.ylabel('Huber Loss')
plt.title('Huber Loss over Iterations')
plt.legend()
plt.show()

# Step 6: Evaluate the model
final_loss = huber_loss(y, y_pred)
print(f"Final Huber Loss: {final_loss}")
