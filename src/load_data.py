import pandas as pd

# Load the dataset
data = pd.read_csv("dataset/healthcare-dataset-stroke-data.csv")

# Remove patient ID
data = data.drop("id", axis=1)

# Calculate the median BMI
median_bmi = data["bmi"].median()

print("Median BMI:", median_bmi)

# Fill missing BMI values with the median
data["bmi"] = data["bmi"].fillna(median_bmi)

# Check missing values again
print("\nMissing values after filling BMI:")
print(data.isnull().sum())
print("\nGender categories:")
print(data["gender"].unique())

print("\nEver married categories:")
print(data["ever_married"].unique())

print("\nWork type categories:")
print(data["work_type"].unique())

print("\nResidence type categories:")
print(data["Residence_type"].unique())

print("\nSmoking status categories:")
print(data["smoking_status"].unique())