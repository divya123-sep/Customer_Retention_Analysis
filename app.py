import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import plotly.figure_factory as ff
import plotly.graph_objects as go

# Load Dataset
@st.cache_data
def load_data():
    # Replace 'customer_churn.csv' with your dataset file
    data = pd.read_csv(r"C:\Users\divya\OneDrive\Documents\App\Customer Churn\Telco-Customer-Churn.csv")
    return data

# Precompute Accuracies
@st.cache_data
def compute_accuracies(X_train, X_test, y_train, y_test):
    accuracies = []
    estimators = range(50, 150, 25)
    for n in estimators:
        temp_model = RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=-1)
        temp_model.fit(X_train, y_train)
        temp_pred = temp_model.predict(X_test)
        accuracies.append(accuracy_score(y_test, temp_pred))
    return list(estimators), accuracies

# App Header
st.title("Customer Churn Prediction App")

# Load Data
data = load_data()

# Prediction Section
st.header("Customer Churn Prediction")

# Data Preprocessing
st.write("Preprocessing the dataset...")
data = pd.get_dummies(data, drop_first=True)  # Encoding categorical variables

# Splitting Data
X = data.drop("Churn_Yes", axis=1)
y = data["Churn_Yes"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
st.write("Training the Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Model Evaluation
st.subheader("Model Performance")
y_pred = model.predict(X_test)
st.write("Accuracy:", accuracy_score(y_test, y_pred))
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# Confusion Matrix
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
confusion_fig = ff.create_annotated_heatmap(cm, x=["No Churn", "Churn"], y=["No Churn", "Churn"], colorscale="Viridis")

# Accuracy Graph
st.subheader("Accuracy Over Different Parameters")
estimators, accuracies = compute_accuracies(X_train, X_test, y_train, y_test)

# Plot the Accuracy Graph
accuracy_fig = go.Figure()
accuracy_fig.add_trace(go.Scatter(x=estimators, y=accuracies, mode='lines+markers', name='Accuracy'))
accuracy_fig.update_layout(
    title="Model Accuracy vs. Number of Estimators",
    xaxis_title="Number of Estimators",
    yaxis_title="Accuracy",
    template="plotly_dark"
)

# Display Side-by-Side Graphs
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(confusion_fig, use_container_width=True)
with col2:
    st.plotly_chart(accuracy_fig, use_container_width=True)
