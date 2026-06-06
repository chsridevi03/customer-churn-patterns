# customer-churn-patterns


📊 Customer Churn Prediction (XGBoost ML Project)

A machine learning project that predicts whether a customer will churn based on behavioral and account data. Built using Python, Pandas, Scikit-learn, and XGBoost, with full preprocessing, training, evaluation, and visualization pipeline.

🚀 Project Overview

Customer churn is a major problem in subscription-based businesses.
This project uses a supervised ML model to predict churn probability and identify key factors influencing customer retention.

🧠 Features

📦 Synthetic dataset generation (for testing/demo)
🧹 Data preprocessing using ColumnTransformer
🔢 Feature scaling using StandardScaler
🏷️ One-hot encoding for categorical variables
⚖️ Class imbalance handling using scale_pos_weight
🌲 XGBoost classification model

📊 Evaluation metrics:
Accuracy
Precision / Recall / F1-score
ROC-AUC score
📈 Visualizations:
Feature importance graph
ROC curve
🛠️ Tech Stack
Python 🐍
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
XGBoost

📂 Project Structure

customer-churn-project/
│
├── customer.py              # Main Python script
├── requirements.txt        # Dependencies
├── README.md               # Project documentation
└── screenshots/            # Output graphs (optional)

⚙️ Installation & Setup

1️⃣ Clone the repository
git clone https://github.com/chsridevi03/customer-churn-project.git
cd customer-churn-project

2️⃣ Create virtual environment
python -m venv venv

Activate:

Windows:

venv\Scripts\activate
3️⃣ Install dependencies
pip install -r requirements.txt
📌 requirements.txt
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
▶️ How to Run
python customer.py

📊 Output

After running the script, you will get:

Model training logs
Classification report
ROC-AUC score
Feature importance graph 📊
ROC curve 📈
📌 Key Insights
Contract type is a major churn indicator
Higher monthly charges increase churn probability
Tenure strongly reduces churn risk
XGBoost performs well on imbalanced datasets
📷 Sample Output

(Add screenshots here in GitHub for better presentation)

Feature Importance Graph
ROC Curve
🎯 Future Improvements
Use real telecom dataset (Kaggle)
Deploy using Streamlit / Flask
Add real-time prediction UI
Improve model accuracy with hyperparameter tuning
👨‍💻 Author

Sridevi
Engineering Student 🚀
Interested in AI, ML, IoT & Smart Systems
