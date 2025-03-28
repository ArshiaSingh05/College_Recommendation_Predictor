import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib


data = pd.read_csv("cleaned_data.csv")
def categorize_rating(rating):
    if rating >= 4.5:
        return "Excellent"
    elif rating >= 3.5:
        return "Good"
    elif rating >= 2.5:
        return "Average"
    else:
        return "Poor"
data["Category"] = data["Overall Rating"].apply(categorize_rating)
X = data[['Average Rating', 'Placement vs Fee Ratio', 'UG fee (tuition fee)', 'PG fee']]
y = data["Category"].astype("category").cat.codes  # Convert to numerical labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#RandomForest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred_train = rf_model.predict(X_train)
rf_pred_test = rf_model.predict(X_test)
# Train GradientBoostingClassifier
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)
gb_pred_train = gb_model.predict(X_train)
gb_pred_test = gb_model.predict(X_test)
#new dataset with predictions
stacked_train = pd.DataFrame({"RF_Pred": rf_pred_train, "GB_Pred": gb_pred_train})
stacked_test = pd.DataFrame({"RF_Pred": rf_pred_test, "GB_Pred": gb_pred_test})
#Meta-Model (Logistic Regression)
meta_model = LogisticRegression()
meta_model.fit(stacked_train, y_train)
final_pred = meta_model.predict(stacked_test)
# Evaluate
accuracy = accuracy_score(y_test, final_pred)
print(f"Final Stacked Model Accuracy: {accuracy:.2f}")

joblib.dump(rf_model, 'random_forest.pkl')
joblib.dump(gb_model, 'gradient_boosting.pkl')
joblib.dump(meta_model, 'meta_model.pkl')
print("Models saved successfully.")
