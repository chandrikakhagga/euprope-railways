# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 19:59:37 2022

@author: LENOVO
"""


import uvicorn
from fastapi import FastAPI
import pickle

app = FastAPI()

@app.get('/')
def home():
    return{'text' :'Price prediction'}


'S.No', 'Train_Names', 'company_name', 'country_name', 'engine_type',
       'Train_Type', 'Train_bogies', 'speed_of_train', 'source_station',
       'destination_station', 'carry_weight', 'maintance_cost', 'Hours',
       'Price']
    
    
    
@app.get('/predict')
def predict(
    engine_type: int,
    Train_Type: int,
    Train_bogies: int,
    speed_of_train: int,
    source_station: int,
    destination_station: int,
    carry_weight: int,
    maintance_cost : int,
    Hours: int):
    
    model = pickle.load(open("G:/data science/360PROJECT_2/venky_duplicate/data_model_p2.pkl","rb"))
    
    makeprediction = model.predict([[engine_type,       Train_Type, Train_bogies, speed_of_train, source_station,
       destination_station, carry_weight, maintance_cost, Hours]])
    output = round(makeprediction[0])
    return {'Price for transport: {}'.format(output) + ' INR'} 

if __name__ == '__main__':
    uvicorn.run(app)

