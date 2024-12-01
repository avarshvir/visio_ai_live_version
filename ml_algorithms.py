import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

#st.set_page_config(page_title="ML Algorithms", layout="wide")

# Handle query parameters to check navigation origin


# Function to check if the target variable is categorical or continuous
def is_categorical(target):
    return target.dtype == 'object' or len(target.unique()) < 20

# Function to display regression metrics
def display_regression_metrics(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    st.write(f"Mean Squared Error: {mse:.2f}")
    st.write(f"Root Mean Squared Error: {rmse:.2f}")
    st.write(f"R-squared: {1 - mse / np.var(y_test):.2f}")

# Function to display classification metrics
def display_classification_metrics(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Accuracy: {accuracy:.2f}")
    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred))

# Load Dataset
st.title('Visio-AI ML Algorithms')
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Drop unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    st.write("Dataset Preview", df.head())

    # Select target variable
    target_variable = st.selectbox("Select Target Variable", df.columns)
    X = df.drop(columns=[target_variable])
    y = df[target_variable]

    # Check if target is categorical or continuous
    if is_categorical(y):
        st.write(f"Target variable `{target_variable}` is categorical, performing Classification.")
        algorithms = ['Logistic Regression', 'Decision Tree Classifier', 'Random Forest Classifier', 'K-Nearest Neighbors', 'SVM']
    else:
        st.write(f"Target variable `{target_variable}` is continuous, performing Regression.")
        algorithms = ['Linear Regression', 'Decision Tree Regressor', 'Random Forest Regressor', 'SVR']

    # Select algorithm
    algorithm_choice = st.selectbox("Select Algorithm", algorithms)

    # Dynamic Train Size
    train_size = st.slider("Train size", 0.1, 0.9, 0.8)  # From 10% to 90%
    test_size = 1 - train_size  # Calculate test size dynamically

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, train_size=train_size, random_state=42)

    # Display Train and Test Dataset Previews
    st.write("Training Data Preview", X_train.head())
    st.write("Test Data Preview", X_test.head())

    # Scale the features if required (e.g., for SVM, KNN, etc.)
    if algorithm_choice in ['SVM', 'K-Nearest Neighbors', 'SVR']:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # Train and Test Model based on selected algorithm
    if algorithm_choice == 'Logistic Regression' and is_categorical(y):
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        display_classification_metrics(y_test, y_pred)

    elif algorithm_choice == 'Decision Tree Classifier' and is_categorical(y):
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        display_classification_metrics(y_test, y_pred)

    elif algorithm_choice == 'Random Forest Classifier' and is_categorical(y):
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        display_classification_metrics(y_test, y_pred)

    elif algorithm_choice == 'K-Nearest Neighbors' and is_categorical(y):
        model = KNeighborsClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        display_classification_metrics(y_test, y_pred)

    elif algorithm_choice == 'SVM' and is_categorical(y):
        model = SVC(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        display_classification_metrics(y_test, y_pred)

    elif algorithm_choice == 'Linear Regression' and not is_categorical(y):
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        display_regression_metrics(y_test, y_pred)

    elif algorithm_choice == 'Decision Tree Regressor' and not is_categorical(y):
        model = DecisionTreeRegressor(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        display_regression_metrics(y_test, y_pred)

    elif algorithm_choice == 'Random Forest Regressor' and not is_categorical(y):
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        display_regression_metrics(y_test, y_pred)

    elif algorithm_choice == 'SVR' and not is_categorical(y):
        model = SVR()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        display_regression_metrics(y_test, y_pred)

    # Visualize predictions vs true values for regression
    if not is_categorical(y):
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
        ax.set_xlabel("True Values")
        ax.set_ylabel("Predictions")
        ax.set_title("True vs Predicted")
        st.pyplot(fig)

    # Visualize confusion matrix for classification
    if is_categorical(y):
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)

    # Input new data points for prediction
    st.subheader("Input New Data for Prediction")
    input_data = {}
    for feature in X.columns:
        input_data[feature] = st.text_input(f"Enter value for {feature}")
    
    if st.button("Get Prediction"):
        try:
            input_data = {k: float(v) for k, v in input_data.items()}
            input_df = pd.DataFrame(input_data, index=[0])
            
            # Scale the input data if necessary
            if algorithm_choice in ['SVM', 'K-Nearest Neighbors', 'SVR']:
                input_df = scaler.transform(input_df)
            
            # Make prediction
            prediction = model.predict(input_df)
            st.write("Predicted Value: ", prediction)
        except ValueError:
            st.write("Please enter valid numerical values for all features.")
