import pandas as pd

# Load your original UNSW-NB15 dataset (before preprocessing)
data = pd.read_csv('C:\\Users\\saish\\OneDrive\\Desktop\\botnet\\UNSW_NB15_testing-set.csv')

# Drop rows with missing values in attack_cat or rate
data = data.dropna(subset=['attack_cat', 'rate'])

# Print unique values
print("Unique attack categories:")
print(data['attack_cat'].unique())

# Group by attack name and see which 'rate' values they map to
print("\nAttack name to rate value mapping:")
print(data.groupby('attack_cat')['rate'].unique())
