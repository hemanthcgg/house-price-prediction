import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from load_data import load_data
from preprocess import preprocess_data

# Load data
df = load_data("data/train.csv")

# Preprocess
df = preprocess_data(df)

X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "models/model.pkl")

print("Model trained and saved successfully")
