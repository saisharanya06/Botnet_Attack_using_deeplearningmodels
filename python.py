import pandas as pd

# Load the original dataset
df = pd.read_csv("C:\\Users\\saish\\OneDrive\\Desktop\\botnet\\UNSW_NB15_testing-set.csv", dtype=str)

# Drop rows with missing attack categories
df.dropna(subset=["attack_cat"], inplace=True)

# Map label column using attack_cat: Normal = 0, Attack = 1
df["label"] = df["attack_cat"].apply(lambda x: 0 if x.lower() == "normal" else 1)

# Select only necessary columns for your model
columns_to_keep = ['sbytes', 'dbytes', 'rate', 'dinpkt', 'tcprtt', 
                   'synack', 'ackdat', 'smean', 'dmean', 'attack_cat']

# Ensure all columns exist in DataFrame
columns_to_keep = [col for col in columns_to_keep if col in df.columns]

# Keep only selected columns
df = df[columns_to_keep]

# Display info for confirmation
print("Dataset after keeping selected columns:")
print(df.head())
print("\nClass Distribution:\n", df["attack_cat"].value_counts())
# print("\nLabel Mapping:\n", df["label"].value_counts())

# Save the preprocessed dataset
df.to_csv("preprocessed_unsw_nb15.csv", index=False)
print("\n✅ Preprocessed dataset saved as 'preprocessed_unsw_nb15.csv'")
