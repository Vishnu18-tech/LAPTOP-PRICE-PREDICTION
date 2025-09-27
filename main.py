import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv('laptop_data.csv')
data.drop(columns=['Unnamed: 0'],inplace=True)

data['Ram'] = data['Ram'].str.replace('GB','')
data['Weight'] = data['Weight'].str.replace('kg','')
data['Ram']=data['Ram'].astype('int32')
data['Weight']=data['Weight'].astype('float32')

data['Touchscreen']=data['ScreenResolution'].apply(lambda x : 1 if 'Touchscreen' in x else 0)
data['Ips']=data['ScreenResolution'].apply(lambda x : 1 if 'IPS' in x else 0)

new = data['ScreenResolution'].str.split('x',n=1,expand=True)
data['X_res'] = new[0]
data['Y_res'] = new[1]
data['X_res'] = data['X_res'].str.replace(',','').str.findall(r'(\d+\.?\d+)').apply(lambda x:x[0])

data['X_res']=data['X_res'].astype('int32')
data['Y_res']=data['X_res'].astype('int32')
data['PPI']=(((data['X_res']**2)+(data['Y_res']**2))**0.5/data['Inches']).astype('float')

data.drop(columns=['ScreenResolution'],inplace=True)
data.drop(columns=['X_res','Y_res','Inches'],inplace=True)

def fetchprocessor(text):
    if text == 'Intel Core i7' or text == 'Intel Core i5' or text == 'Intel Core i3':
        return text
    else:
        if text.split()[0] == 'Intel':
            return 'Other Intel Processor'
        else:
            return 'AMD Processor'
data['Cpu Name'] = data['Cpu'].apply(lambda x:" ".join(x.split()[0:3]))
data['Cpu Brand']=data['Cpu Name'].apply(fetchprocessor)

data.drop(columns=['Cpu Name','Cpu'],inplace=True)
data['Memory'] = data['Memory'].astype(str).replace(r'\.0', '', regex=True)
new = data['Memory'].str.split('+', n=1, expand=True)
data['first'] = new[0].str.strip()
data['second'] = new[1]

def mem_to_gb(x):
    x = str(x).upper().strip()
    if "TB" in x:  
        num = ''.join(filter(lambda c: c.isdigit() or c=='.', x))
        return int(float(num) * 1024)
    elif "GB" in x:
        num = ''.join(filter(lambda c: c.isdigit() or c=='.', x))
        return int(float(num))
    else:
        num = ''.join(filter(str.isdigit, x))
        return int(num) if num else 0

data['first'] = data['first'].apply(mem_to_gb)
data['second'] = data['second'].fillna("0").apply(mem_to_gb)
data["Layer1HDD"] = new[0].apply(lambda x: 1 if "HDD" in str(x).upper() else 0)
data["Layer1SSD"] = new[0].apply(lambda x: 1 if "SSD" in str(x).upper() else 0)
data["Layer1Hybrid"] = new[0].apply(lambda x: 1 if "HYBRID" in str(x).upper() else 0)
data["Layer1Flash_Storage"] = new[0].apply(lambda x: 1 if "FLASH" in str(x).upper() else 0)

data["Layer2HDD"] = new[1].apply(lambda x: 1 if "HDD" in str(x).upper() else 0 if pd.notna(x) else 0)
data["Layer2SSD"] = new[1].apply(lambda x: 1 if "SSD" in str(x).upper() else 0 if pd.notna(x) else 0)
data["Layer2Hybrid"] = new[1].apply(lambda x: 1 if "HYBRID" in str(x).upper() else 0 if pd.notna(x) else 0)
data["Layer2Flash_Storage"] = new[1].apply(lambda x: 1 if "FLASH" in str(x).upper() else 0 if pd.notna(x) else 0)

data["HDD"] = data["first"]*data["Layer1HDD"] + data["second"]*data["Layer2HDD"]
data["SSD"] = data["first"]*data["Layer1SSD"] + data["second"]*data["Layer2SSD"]
data["Hybrid"] = data["first"]*data["Layer1Hybrid"] + data["second"]*data["Layer2Hybrid"]
data["Flash_Storage"] = data["first"]*data["Layer1Flash_Storage"] + data["second"]*data["Layer2Flash_Storage"]
data.drop(columns=['first', 'second', 'Layer1HDD', 'Layer1SSD', 'Layer1Hybrid',
                   'Layer1Flash_Storage', 'Layer2HDD', 'Layer2SSD', 'Layer2Hybrid',
                   'Layer2Flash_Storage'], inplace=True)

data.drop(columns=['Memory','Hybrid','Flash_Storage'],inplace=True)
data['Gpu Brand']=data['Gpu'].apply(lambda x : x.split()[0])
data.drop(columns=['Gpu'],inplace=True)

def cat_os(inp):
    if inp == 'Windows 10' or inp == 'Windows 7' or inp == 'Windows 10 S':
        return 'Windows'
    elif inp == 'macOS' or inp == 'Mac OS X':
        return 'Mac'
    else:
        return 'Others/No OS/Linux'
data['Os']=data['OpSys'].apply(cat_os)
data.drop(columns=['OpSys','Weight','PPI'],inplace=True)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

X=data.drop(columns=['Price'])
y=np.log(data['Price'])

categorical=['Company','TypeName','Cpu Brand','Gpu Brand','Os']
numerical=['Ram','Touchscreen','Ips','HDD','SSD']

preprocessing = ColumnTransformer(
    transformers=[
        ('num',StandardScaler(),numerical),
        ('cat',OneHotEncoder(),categorical)
    ])

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=Pipeline(steps=[
    ('preprocessor',preprocessing),
    ('regressor',LinearRegression())
])

model.fit(X_train,y_train)

y_pred=model.predict(X_test)

y_pred_original=np.exp(y_pred)
y_test_original=np.exp(y_test)

MSE=mean_squared_error(y_pred_original,y_test_original)
print("MSE : ",MSE)
print("RMSE : ",np.sqrt(MSE))
print("R2 Score : ",r2_score(y_pred_original,y_test_original))
#print(data)
new_laptop=pd.DataFrame([{
    'Company':'HP',
    'TypeName':'Notebook',
    'Ram':8,
    'Touchscreen':0,
    'Ips':0,
    'Cpu Brand':'Intel Core i5',
    'HDD':0,
    'SSD':256,
    'Gpu Brand':'Intel',
    'Os':'Windows'
}])
predicted_log_price=model.predict(new_laptop)
predicted_price=np.exp(predicted_log_price[0])
print(predicted_price)

plt.scatter(y_test_original, y_pred_original)
plt.xlabel("ACTUAL PRICES",fontsize=18,fontweight='bold')
plt.ylabel("PREDICTED PRICES",fontsize=18,fontweight='bold')
plt.title("ACTUAL VS PREDICTED PRICES",fontsize=18,fontweight='bold')
plt.plot([min(y_test_original), max(y_test_original)], 
         [min(y_test_original), max(y_test_original)], 'r--')
plt.show()

from joblib import dump
dump(model, "laptop_price_model.pkl")