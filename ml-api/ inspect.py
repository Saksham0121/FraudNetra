import pandas as pd

df = pd.read_csv("")

print("Shape:", df.shape)

print("\nColumns:\n", df.columns.tolist())

print("\nInfo:\n")
print(df.info())

print("\nFraud count:\n", df["is_fraud"].value_counts())

print("\nMissing values:\n", df.isnull().sum())