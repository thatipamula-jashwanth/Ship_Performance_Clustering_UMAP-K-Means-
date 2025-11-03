import pandas as pd

df = pd.read_csv('Ship_Performance_Dataset.csv')

print(df.info())
print('shape of dataset',df.shape)

print('missing values in dataset', df.isnull().sum())

placeholders = ['Nan', 'nan', 'Unknown', 'UNKNOWN','N/A','']

print('total placeholders', df.isin([placeholders]).sum())

print('unique values for column', df.nunique())

cat_features = df.select_dtypes(include=['object']).columns.tolist
print('categorical columns', cat_features)

