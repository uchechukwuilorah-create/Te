import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

def train_model():
    # Load data
    df = pd.read_csv('/home/ubuntu/upload/Student_performance_data_.csv')
    
    # Drop StudentID as it's just an identifier
    # GPA is highly correlated with GradeClass (usually GradeClass is derived from GPA)
    # If we want to predict GradeClass based on student habits/background, we might drop GPA
    # However, for a general model, let's keep it or decide based on the goal.
    # Let's assume we want to predict GradeClass based on all other factors.
    
    X = df.drop(['StudentID', 'GradeClass'], axis=1)
    y = df['GradeClass']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train model
    # Using RandomForest as it handles categorical and numerical data well
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save the model and the feature names
    joblib.dump(model, 'student_model.pkl')
    joblib.dump(X.columns.tolist(), 'feature_names.pkl')
    print("\nModel and feature names saved successfully.")

if __name__ == "__main__":
    train_model()
