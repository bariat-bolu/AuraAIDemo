import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_curve, auc

# Title and description for the Streamlit app
st.title("Migraine Prediction App")
st.write("This application predicts whether a person will have a migraine based on various health factors.")

# Step 1: Upload data (for demo purposes, use placeholder data)
@st.cache
def load_data():
    # Example: Replace with loading your actual dataset
    # df = pd.read_csv('your_dataset.csv')
    # For the demo, create some dummy data:
    df = pd.DataFrame({
        'age': np.random.randint(18, 80, size=1000),
        'bmi': np.random.uniform(15.0, 40.0, size=1000),
        'glucose': np.random.uniform(50.0, 200.0, size=1000),
        'migraine': np.random.choice([0, 1], size=1000)
    })
    return df

data = load_data()

# Step 2: Feature Engineering and Preprocessing
def preprocess_data(df):
    X = df.drop('migraine', axis=1)
    y = df['migraine']
    
    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Handle class imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_scaled, y)
    
    # Apply PCA
    pca = PCA(n_components=0.95)
    X_res_pca = pca.fit_transform(X_res)
    
    return X_res_pca, y_res, pca, scaler

X_res_pca, y_res, pca, scaler = preprocess_data(data)

# Step 3: Train-test split
X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_res_pca, y_res, test_size=0.2, random_state=42)

# Step 4: Model Training and Tuning
def train_model(X_train, y_train):
    rf_model = RandomForestClassifier(random_state=42)
    param_dist_rf = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True, False]
    }
    
    random_search_rf = RandomizedSearchCV(rf_model, param_distributions=param_dist_rf, n_iter=10, cv=3, n_jobs=-1, verbose=2, random_state=42)
    random_search_rf.fit(X_train, y_train)
    
    return random_search_rf.best_estimator_

best_rf_model = train_model(X_train_pca, y_train)

# Step 5: Prediction
def make_prediction(model, X, threshold=0.3):
    y_pred_prob = model.predict_proba(X)[:, 1]
    y_pred_adjusted = (y_pred_prob >= threshold).astype(int)
    return y_pred_adjusted, y_pred_prob

y_pred_adjusted_rf, y_pred_prob_rf = make_prediction(best_rf_model, X_test_pca)

# Step 6: Display Results
# User input form
st.sidebar.header("Enter Patient's Information")
def user_input_features():
    age = st.sidebar.slider("Age", 18, 80, 30)
    bmi = st.sidebar.slider("BMI", 15.0, 40.0, 25.0)
    glucose = st.sidebar.slider("Glucose Level", 50.0, 200.0, 100.0)
    data = {'age': age, 'bmi': bmi, 'glucose': glucose}
    features = pd.DataFrame(data, index=[0])
    return features

input_data = user_input_features()

# Preprocess user input
input_data_scaled = scaler.transform(input_data)
input_data_pca = pca.transform(input_data_scaled)

# Prediction for user input
input_prediction = best_rf_model.predict(input_data_pca)
input_prob = best_rf_model.predict_proba(input_data_pca)[:, 1]

st.write(f"Prediction: {'Migraine' if input_prediction[0] == 1 else 'No Migraine'}")
st.write(f"Prediction Probability: {input_prob[0]:.2f}")

# Step 7: Confusion Matrix
conf_matrix_rf = confusion_matrix(y_test, y_pred_adjusted_rf)

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Purples', cbar=False,
            xticklabels=['No Migraine', 'Migraine'], yticklabels=['No Migraine', 'Migraine'])
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix (Random Forest - Tuned)')
st.pyplot(fig)

# Step 8: Precision-Recall and ROC Curves
precision_rf, recall_rf, _ = precision_recall_curve(y_test, y_pred_prob_rf)
pr_auc_rf = auc(recall_rf, precision_rf)

fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_prob_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)

# Plot Precision-Recall and ROC Curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Precision-Recall Curve
ax1.plot(recall_rf, precision_rf, color='green', lw=2, label=f'Precision-Recall curve (AUC = {pr_auc_rf:.2f})')
ax1.set_xlabel('Recall')
ax1.set_ylabel('Precision')
ax1.set_title('Precision-Recall Curve')
ax1.legend(loc="lower left")

# ROC Curve
ax2.plot(fpr_rf, tpr_rf, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc_rf:.2f})')
ax2.plot([0, 1], [0, 1], color='gray', linestyle='--')
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('Receiver Operating Characteristic (ROC)')
ax2.legend(loc="lower right")

st.pyplot(fig)

# Step 9: Display classification report
st.write("Classification Report:")
st.text(classification_report(y_test, y_pred_adjusted_rf))
