import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('data/hypertension_dataset.csv')

# ✅ Check for general insight into dataset and null values
print(df.info())
print(df.isnull().sum())

print()
df["Medication"] = df["Medication"].fillna("None")

df["Has_Hypertension"] = df["Has_Hypertension"].map({"Yes": 1, "No": 0})
df["Family_History"] = df["Family_History"].map({"Yes": 1, "No": 0})
df["Smoking_Status"] = df["Smoking_Status"].map({"Smoker": 1, "Non-Smoker": 0})

df = df.drop_duplicates()
df = df.dropna()

print(df.isnull().sum())
print(df.shape)

print(df.describe())
print(df.corr(numeric_only=True))

print(df.groupby('Has_Hypertension').mean(numeric_only=True))

df[df['Has_Hypertension'] == 0]['Age'].hist(alpha=0.5, label='No Hypertension')
df[df['Has_Hypertension'] == 1]['Age'].hist(alpha=0.5, label='Hypertension') 

plt.legend()
plt.title('Age comparison')
plt.show()


df[df['Has_Hypertension'] == 0]['BMI'].hist(alpha=0.5, label='No Hypertension')
df[df['Has_Hypertension'] == 1]['BMI'].hist(alpha=0.5, label='Hypertension') 

plt.legend()
plt.title('BMI Comparison')
plt.show()


df.groupby('Family_History')['Has_Hypertension'].mean().plot(kind='bar')
plt.title('Hypertension Rate by Family History')
plt.show()

df.to_csv('data/cleaned_data.csv', index=False)
