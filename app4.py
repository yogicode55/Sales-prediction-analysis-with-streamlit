import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load your dataset
data = pd.read_csv("D:/Datasets/code2/test_data.csv")  # Replace "your_dataset.csv" with your actual dataset file

# Define features and target
features = ['product_identifier', 'department_identifier', 'outlet']
X = data[features]
y = data['category_of_product']

# Train your models
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Specify feature names
lin_reg.feature_names_in_ = features

xgb_reg = XGBRegressor()
xgb_reg.fit(X, y)

# Define a function to make predictions and calculate accuracy
def predict_sales(product_identifier, department_identifier, outlet, model):
    prediction = model.predict([[product_identifier, department_identifier, outlet]])
    return prediction[0]

# Define a function to plot the relationship between features and sales
def plot_relationship(feature, sales, xlabel):
    plt.figure(figsize=(10, 6))
    plt.scatter(feature, sales, color='blue', alpha=0.5)
    plt.xlabel(xlabel)
    plt.ylabel('Sales')
    plt.title(f'Relationship between {xlabel} and Sales')
    plt.grid(True)
    st.pyplot()

# Main function to create the Streamlit app
def main():
    st.title('Sales Prediction Analysis')
    
    # Input fields for product_identifier, department_identifier, and outlet
    product_identifier = st.number_input('Enter Product Identifier:')
    department_identifier = st.number_input('Enter Department Identifier:')
    outlet = st.number_input('Enter Outlet:')
    
    # Choose which model to use
    model_choice = st.radio('Select Model:', ('Linear Regression', 'XGBoost'))
    
    # Make prediction based on selected model
    if model_choice == 'Linear Regression':
        prediction = predict_sales(product_identifier, department_identifier, outlet, lin_reg)
        model = lin_reg
    else:
        prediction = predict_sales(product_identifier, department_identifier, outlet, xgb_reg)
        model = xgb_reg
    
    st.write(f'Predicted Sales: {prediction}')
    
    # Calculate and display accuracy
    actual_sales = data['sales']
    predicted_sales = [predict_sales(pid, dep_id, out, model) for pid, dep_id, out in zip(data['product_identifier'], data['department_identifier'], data['outlet'])]
    mse = mean_squared_error(actual_sales, predicted_sales)
    st.write(f'Mean Squared Error (MSE): {mse}')
    
    # Plot relationship between features and sales
    plot_relationship(data['product_identifier'], data['sales'], 'Product Identifier')
    plot_relationship(data['department_identifier'], data['sales'], 'Department Identifier')
    plot_relationship(data['outlet'], data['sales'], 'Outlet')

if __name__ == '__main__':
    main()
