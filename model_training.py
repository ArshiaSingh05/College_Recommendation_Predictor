import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import pickle

data = pd.read_csv("cleaned_data.csv")
X = data[['Average Rating', 'Placement vs Fee Ratio', 'UG fee (tuition fee)', 'PG fee']]
y = data['Overall Rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R²): {r2}")
y_pred = model.predict(X_test)
#Saving the model to reuse it without retraining
joblib.dump(model, 'trained_model.pkl')
print("Model saved successfully as 'model.pkl'")
# Save the trained model
with open('training_model.pkl', 'wb') as file:
    pickle.dump(model, file)