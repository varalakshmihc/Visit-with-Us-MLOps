
import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset from the repository data folder
data_path = pd.read_csv("/content/drive/MyDrive/Practice/tourism.csv")
df = pd.read_csv(data_path)

# Remove unnecessary columns
# Unnamed: 0 is an index column, and CustomerID is only an identifier.
df = df.drop(columns=["Unnamed: 0", "CustomerID"], errors="ignore")

# Separate features and target
X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Save the splits locally as CSV files
X_train.to_csv("Xtrain.csv", index=False)
X_test.to_csv("Xtest.csv", index=False)
y_train.to_csv("ytrain.csv", index=False)
y_test.to_csv("ytest.csv", index=False)

print("Data preparation completed successfully.")
print(f"Training features: {X_train.shape}")
print(f"Testing features:  {X_test.shape}")
print(f"Training target:   {y_train.shape}")
print(f"Testing target:    {y_test.shape}")
