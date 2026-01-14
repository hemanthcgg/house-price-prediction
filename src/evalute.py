import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
from load_data import load_data
from preprocess import preprocess_data
from sklearn.model_selection import train_test_split
import numpy as np

df = load_data("data/train.csv")
df = preprocess_data(df)

X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = joblib.load("models/model.pkl")
preds = model.predict(X_test)

mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))

print(f"MAE: {mae}")
print(f"RMSE: {rmse}")
