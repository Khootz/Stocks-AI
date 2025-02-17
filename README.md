# Stocks AI: Predicting Stock Movements with LSTM

This project aims to predict stock price movements using historical stock data and a Long Short-Term Memory (LSTM) model, a type of recurrent neural network that is particularly good at learning from sequential data.

## Project Overview

I collected historical stock data for various ETFs and stocks from Kaggle, spanning from 1999 to 2017. The data includes the following columns: 
- **Date**
- **Open**
- **High**
- **Low**
- **Volume**

### Data Cleaning & Preparation:
- Filtered out non-stock data and focused on stocks from Yahoo Finance API.
- Performed web scraping using Python to assign stocks to their respective industries.
- Filtered stocks by industry ranking (top-ranked sectors by number of stocks).
- Focused on the **Biomedical** industry, combining data from multiple stocks into a single dataset.

### Feature Engineering:
- **Momentum Analysis**: Calculated the 10-day and 50-day moving averages for each stock price.
- The resulting dataset contains the following features:
  - Date
  - Open, High, Low, Volume
  - 10-day moving average
  - 50-day moving average

## Model: LSTM (Long Short-Term Memory)
LSTM is an advanced type of recurrent neural network (RNN) that works well for sequential data, such as stock prices over time. It learns from past time steps and captures short-term patterns to predict future trends.

### Model Workflow:
- **Data Split**: Split the data into training and testing datasets.
- **Training**: Trained the LSTM model to predict whether the price of a stock will increase or decrease the next day (classification problem).
- **Testing**: Evaluated the model on a test set, achieving a **60% accuracy** in predicting stock price movements.

## Libraries Used

- Python 3.x
- TensorFlow / Keras (for LSTM model)
- Pandas
- Numpy
- Matplotlib (for data visualization)
- Yahoo Finance API for web scraping

