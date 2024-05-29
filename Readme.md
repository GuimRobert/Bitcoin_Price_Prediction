# Bitcoin Price Prediction Using LSTM

## Project Overview
This project aims to predict the next-day Bitcoin price using machine learning techniques, specifically focusing on a Long Short-Term Memory (LSTM) model. The project includes data preparation, model training, and evaluation to forecast Bitcoin prices based on historical data.

## Data Source
The data used in this project is sourced from Yahoo Finance using the `yfinance` library. The dataset includes the following features:
- Date
- Open
- High
- Low
- Close
- Volume

## Data Preparation
- Historical Bitcoin data is fetched and cleaned to remove unnecessary columns.
- The data is then normalized using the `MinMaxScaler` to aid in the training of the LSTM model.
- The dataset is split into training and testing sets, with 80% of the data used for training.

## Model Description
The LSTM model used in this project includes:
- Four LSTM layers with dropout regularization to prevent overfitting.
- The model is compiled using the Adam optimizer and Mean Squared Error (MSE) loss function.
- Early stopping and learning rate reduction techniques are employed to optimize training.

## Training
- The model is trained on the historical closing prices of Bitcoin.
- A look-back period of 600 days is used to prepare the training and testing datasets.

## Evaluation
- The model's performance is evaluated based on its directional accuracy, which measures the model's ability to predict the direction of price movement.
- The LSTM model achieved a directional accuracy of up to 58.25% in various trials.

## Conclusion
- The LSTM model shows potential in predicting Bitcoin prices with reasonable accuracy.
- Continuous refinement and adjustment of model parameters can enhance its predictive capabilities.

## Usage
To run this project, ensure you have Python installed along with the necessary libraries:
- numpy
- matplotlib
- seaborn
- pandas
- yfinance
- keras

Clone the repository, navigate to the project directory, and run the Jupyter notebook to replicate the analysis and model training.

## Future Work
- Experiment with different model architectures and hyperparameters.
- Incorporate additional features such as macroeconomic indicators to improve the model's accuracy.
- Deploy the model as a web application for real-time Bitcoin price prediction.
