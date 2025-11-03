import pandas as pd 
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('ship_performance_cleaned.csv')

df = pd.get_dummies(df,columns=['Ship_Type','Route_Type','Engine_Type','Maintenance_Status','Weather_Condition'], drop_first=True, dtype=int)

df.to_csv('ship_performance_encoded.csv',index=False)
print('\n dataset encoded and saved as ship_performence_encoded.csv')