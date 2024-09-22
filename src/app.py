from utils import db_connect
engine = db_connect()

# your code here

# app.py

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest
from pickle import dump
import matplotlib.pyplot as plt

# Step 1: Load the Dataset
DATA_URL = "https://raw.githubusercontent.com/4GeeksAcademy/decision-tree-project-tutorial/main/diabetes.csv"

def load_data(url):
    data = pd.read_csv(url)
    return data.drop_duplicates().reset_index(drop=True)

# Step 2: Exploratory Data Analysis (EDA) and Train/Test Split
def preprocess_data(data):
    X = data.drop("Outcome", axis=1)
    y = data["Outcome"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Feature selection
    selector = SelectKBest(k=7)
    selector.fit(X_train, y_train)
    selected_columns = X_train.columns[selector.get_support()]
    
    X_train_sel = pd.DataFrame(selector.transform(X_train), columns=selected_columns)
    X_test_sel = pd.DataFrame(selector.transform(X_test), columns=selected_columns)
    
    return X_train_sel, X_test_sel, y_train, y_test

# Step 3: Train Decision Tree Classifier
def train_decision_tree(X_train, y_train):
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

# Step 4: Optimize Decision Tree with GridSearch
def optimize_model(model, X_train, y_train):
    hyperparams = {
        "criterion": ["gini", "entropy"],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    }

    grid = GridSearchCV(model, hyperparams, scoring="accuracy", cv=10)
    grid.fit(X_train, y_train)

    print(f"Best hyperparameters: {grid.best_params_}")
    return grid.best_estimator_

# Step 5: Evaluate the Model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy * 100:.2f}%")
    return accuracy

# Step 6: Visualize the Decision Tree
def visualize_tree(model, feature_names):
    fig = plt.figure(figsize=(15,15))
    tree.plot_tree(model, feature_names=feature_names, class_names=["0", "1"], filled=True)
    plt.show()

# Step 7: Save the Model
def save_model(model, filename):
    dump(model, open(filename, "wb"))

# Main Function
if __name__ == "__main__":
    # Load and preprocess the data
    data = load_data(DATA_URL)
    X_train, X_test, y_train, y_test = preprocess_data(data)
    
    # Train the decision tree model
    model = train_decision_tree(X_train, y_train)
    
    # Optimize the model using grid search
    best_model = optimize_model(model, X_train, y_train)
    
    # Evaluate the optimized model
    evaluate_model(best_model, X_test, y_test)
    
    # Visualize the decision tree
    visualize_tree(best_model, X_train.columns)
    
    # Save the model
    save_model(best_model, "../models/optimized_diabetes_model.sav")
