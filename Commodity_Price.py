import uvicorn
from fastapi import FastAPI
from base import var_data_1
from meta import var_data_2
from base2 import var_data_3
from fastapi.middleware.cors import CORSMiddleware
import pickle


app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
   allow_origins=origins,
    allow_credentials = True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# loading the model --
pickle_in = open("gold_price_prediction.pkl","rb") # model1 loaded
regressor = pickle.load(pickle_in)

pickle_in = open("Car_Prediction.pkl","rb")  # model2 loaded
lass_reg_model = pickle.load(pickle_in)

pickle_in = open("Bike_Price.pkl","rb")  # model3 loaded
lr = pickle.load(pickle_in) 

@app.get('/')
def index():
    return {'Deployment':'Hello and Welcome to the Commodity Price Prediction API'}


@app.post('/predict_1')
def Gold_Prediction(data:var_data_1):
    data = data.dict()
    SPX = data['SPX']
    USO = data['USO'] # VALUES INPUT HAS GIVEN 
    SLV = data['SLV']
    EUR = data['EUR']

    prediction = regressor.predict([[SPX,USO,SLV,EUR]])
    return {
        'prediction': prediction.tolist()
    }

@app.post('/predict_2')
def Car_Price(data:var_data_2):
    data = data.dict()
    Year = data['Year']
    Present_Price =  data['Present_Price']
    Kms_Driven = data['Kms_Driven']
    Fuel_Type = data['Fuel_Type']
    Seller_Type = data['Seller_Type']
    Transmission = data['Transmission']
    Owner = data['Owner']

    prediction = lass_reg_model.predict([[Year,Present_Price,Kms_Driven,Fuel_Type,Seller_Type,Transmission,Owner]])

    return {
        'prediction' : prediction.tolist()
    }


@app.post('/predict_3')
def Bike_Price(data:var_data_3):
    data = data.dict()
    Year = data['Year']
    Seller_Type = data['Seller_Type']
    Owner = data['Owner']
    KM_Driven = data['KM_Driven']
    Ex_Showroom_Price = data['Ex_Showroom_Price']

    prediction = lr.predict([[Year,Seller_Type,Owner,KM_Driven,Ex_Showroom_Price]])

    return {
        'prediction' : prediction.tolist()
    }

if __name__=='__main__':
    uvicorn.run(app,host='127.0.0.1',port=5000)
