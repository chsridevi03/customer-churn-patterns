import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from xgboost import XGBClassifier

# ==========================================
# 1. GENERATE SYNTHETIC DATA (For Testing)
# ==========================================
# Replace this section with: df = pd.read_csv("your_data.csv")
def load_sample_data():
    np.random.seed(42)
    n_samples = 2000
    
    data = {
        'CustomerID': range(1001, 1001 + n_samples),
        'Age': np.random.randint(18, 70, size=n_samples),
        'Tenure': np.random.randint(1, 72, size=n_samples), # months
        'MonthlyCharges': np.random.uniform(20, 120, size=n_samples),
        'TotalCharges': np.random.uniform(100, 5000, size=n_samples),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], size=n_samples, p=[0.5, 0.3, 0.2]),
        'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], size=n_samples),
        'Churn': np.random.choice([0, 1], size=n_samples, p=[0.75, 0.25]) # 25% churn rate
    }
    return pd.DataFrame(data)

print("🚀 Loading data...")
df = load_sample_data()

# ==========================================
# 2. DATA PREPROCESSING & FEATURE ENGINEERING
# ==========================================
print("🧹 Preprocessing data...")

# Drop identifier columns not useful for prediction
X = df.drop(columns=['CustomerID', 'Churn'])
y = df['Churn']

# Define categorical and numerical features
num_features = ['Age', 'Tenure', 'MonthlyCharges', 'TotalCharges']
cat_features = ['Contract', 'PaymentMethod']

# Create a preprocessor using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_features)
    ])

# Split data into train and test sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Transform the datasets
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Get feature names after one-hot encoding for analysis later
cat_encoder = preprocessor.named_transformers_['cat']
encoded_cat_features = list(cat_encoder.get_feature_names_out(cat_features))
all_features = num_features + encoded_cat_features

# ==========================================
# 3. MODEL TRAINING (XGBoost)
# ==========================================
print("🏋️ Training XGBoost Model...")

# scale_pos_weight accounts for the class imbalance (75% stay vs 25% churn)
scale_pos_weight_value = (len(y_train) - sum(y_train)) / sum(y_train)

model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight_value,
    random_state=42,
    eval_metric='logloss'
)

model.fit(X_train_transformed, y_train)

# ==========================================
# 4. EVALUATION & METRICS
# ==========================================
print("\n📊 EVALUATION REPORT:")
y_pred = model.predict(X_test_transformed)
y_prob = model.predict_proba(X_test_transformed)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

auc_score = roc_auc_score(y_test, y_prob)
print(f"ROC-AUC Score: {auc_score:.4f}")

# ==========================================
# 5. VISUALIZATION (Feature Importance & ROC)
# ==========================================
plt.figure(figsize=(15, 5))

# Plot 1: Feature Importance
plt.subplot(1, 2, 1)
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
sns.barplot(x=importances[indices], y=np.array(all_features)[indices], palette="viridis")
plt.title("Key Drivers of Customer Churn (Feature Importance)")
plt.xlabel("Importance Score")

# Plot 2: ROC Curve
plt.subplot(1, 2, 2)
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")

plt.tight_layout()
print("\n📉 Displaying Analysis Plots...")
plt.show()