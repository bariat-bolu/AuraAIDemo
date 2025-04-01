from sklearn.decomposition import PCA
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Step 1: Load environmental and migraine data (assuming you have them as DataFrames or arrays)
# Example DataFrames (replace with actual loading code if necessary)
environmental_data = pd.DataFrame({
    'Light Intensity': light_intensity,  # Replace with actual data
    'Noise Levels': noise_levels,        # Replace with actual data
    'Air Pressure': air_pressure         # Replace with actual data
})

migraine_data = pd.DataFrame({
    'Heart Rate': heart_rate,
    'Skin Temp': skin_temp,
    'EDA': eda,
    'PPG': ppg,
    'Migraine Intensity': migraine_intensity,
    'Migraine Duration (min)': migraine_duration,
    'Migraine Frequency': migraine_frequency,
    'Nausea': nausea,
    'Vomit': vomit,
    'Phonophobia': phonophobia,
    'Photophobia': photophobia,
    'Age': age,
    'Timestamp': timestamps   # Add the timestamp column here
})

# Combine the environmental and migraine data into one DataFrame
combined_data = pd.concat([environmental_data, migraine_data], axis=1)

# Step 2: Drop the 'Timestamp' column (since it's not needed for scaling)
X = combined_data.drop(columns=['Migraine Intensity', 'Timestamp'])

# Step 3: Convert 'Migraine Intensity' to binary classes (1 = migraine, 0 = no migraine)
threshold = 1  # Define a threshold for categorizing migraine intensity
y = (combined_data['Migraine Intensity'] > threshold).astype(int)

# Step 4: Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_scaled, y)

# Step 6: Apply PCA for dimensionality reduction (optional)
pca = PCA(n_components=0.95)  # Retaining 95% of variance
X_res_pca = pca.fit_transform(X_res)
print(f"Original feature count: {X.shape[1]}, PCA feature count: {X_res_pca.shape[1]}")

# Step 7: Split the data into training and test sets
X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_res_pca, y_res, test_size=0.2, random_state=42)

# Step 8: Model Tuning with RandomizedSearchCV
param_dist_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True, False]
}

# RandomForestClassifier
rf_model = RandomForestClassifier(random_state=42)

# RandomizedSearchCV to tune the Random Forest
random_search_rf = RandomizedSearchCV(rf_model, param_distributions=param_dist_rf, n_iter=10, cv=3, n_jobs=-1, verbose=2, random_state=42)
random_search_rf.fit(X_train_pca, y_train)

# Get the best model from random search
best_rf_model = random_search_rf.best_estimator_

# Step 9: Train the model on the PCA-transformed training data
best_rf_model.fit(X_train_pca, y_train)

# Step 10: Predict on the transformed test data
y_pred_rf = best_rf_model.predict(X_test_pca)
y_pred_prob_rf = best_rf_model.predict_proba(X_test_pca)[:, 1]

# Step 11: Adjust the decision threshold to improve recall for class 1 (migraine)
threshold = 0.3  # Lower the threshold to increase recall
y_pred_adjusted_rf = (y_pred_prob_rf >= threshold).astype(int)

# Step 12: Model Evaluation
print(f"Accuracy: {accuracy_score(y_test, y_pred_adjusted_rf):.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_adjusted_rf))

# Confusion Matrix
conf_matrix_rf = confusion_matrix(y_test, y_pred_adjusted_rf)

# Plot Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Purples', cbar=False,
            xticklabels=['No Migraine', 'Migraine'], yticklabels=['No Migraine', 'Migraine'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Random Forest - Tuned)')
plt.show()

# Precision-Recall and ROC Curve Evaluation
precision_rf, recall_rf, _ = precision_recall_curve(y_test, y_pred_prob_rf)
pr_auc_rf = auc(recall_rf, precision_rf)
print(f"Precision-Recall AUC: {pr_auc_rf:.2f}")

fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_prob_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)
print(f"ROC AUC: {roc_auc_rf:.2f}")

# Plot Precision-Recall and ROC Curves
plt.figure(figsize=(10, 5))

# Precision-Recall Curve
plt.subplot(1, 2, 1)
plt.plot(recall_rf, precision_rf, color='green', lw=2, label=f'Precision-Recall curve (AUC = {pr_auc_rf:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")

# ROC Curve
plt.subplot(1, 2, 2)
plt.plot(fpr_rf, tpr_rf, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc_rf:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")

plt.tight_layout()
plt.show()

# Cross-validation to check model stability
cv_scores = cross_val_score(best_rf_model, X_res_pca, y_res, cv=5, scoring='accuracy')  # 5-fold cross-validation
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean cross-validation accuracy: {cv_scores.mean():.2f}")

# Final model evaluation on the test set
X_train, X_test, y_train, y_test = train_test_split(X_res_pca, y_res, test_size=0.2, random_state=42)
best_rf_model.fit(X_train, y_train)
y_pred_test = best_rf_model.predict(X_test)

# Accuracy on the final test set
test_accuracy = accuracy_score(y_test, y_pred_test)
print(f"Test set accuracy: {test_accuracy:.2f}")

# Classification Report for final test set
print("Test set Classification Report:")
print(classification_report(y_test, y_pred_test))

# Confusion Matrix for final test set
conf_matrix_test = confusion_matrix(y_test, y_pred_test)

# Plot Confusion Matrix for final test set
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix_test, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['No Migraine', 'Migraine'], yticklabels=['No Migraine', 'Migraine'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Test Set)')
plt.show()
