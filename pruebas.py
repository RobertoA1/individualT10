import pandas as pd

df = pd.read_csv("./datos/titanic.csv")
df.drop(columns=["Name", "Ticket", "Cabin", "Sex"], inplace=True)

print(df.head())

cols = df.columns

for col in df.columns:
    df[col].fillna(df[col].mean(), inplace=True)

print(df.head())