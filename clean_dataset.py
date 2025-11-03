import pandas as pd

df = pd.read_csv('Ship_Performance_Dataset.csv')

print('Shape Before Cleaning',df.shape)

missing_val = df.dropna(inplace=True)
print('removed rows with missing values')
print('shape After Cleaning',df.shape)

df.to_csv('ship_performance_cleaned.csv',index=False)
print('dataset cleaned and saved as ship_performance_cleaned.csv', )

print('Toatl missing values after cleaining',df.isnull().sum().sum())

