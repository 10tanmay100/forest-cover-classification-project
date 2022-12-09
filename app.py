from flask import Flask
import pickle
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin
from matplotlib.pyplot import pcolor
import pandas as pd

app=Flask(__name__,template_folder="templates")
cluster0=pickle.load(open("model_cluster0.pkl","rb"))
cluster1=pickle.load(open("model_cluster1.pkl","rb"))
scaler=pickle.load(open("preprocessed.pkl","rb"))

@app.route('/',methods=['POST', 'GET'])
@cross_origin()
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
@cross_origin()
def predict():
    '''
    For rendering results on HTML GUI
    '''
    a=request.form.get("Aspect")
    b=request.form.get("Elevation")
    c=request.form.get("Hillshade_9am")
    d=request.form.get("Hillshade_Noon")
    e=request.form.get("Horizontal_Distance_To_Fire_Points")
    f=request.form.get("Horizontal_Distance_To_Hydrology")
    g=request.form.get("Horizontal_Distance_To_Roadways")
    x=request.form.get("Vertical_Distance_To_Hydrology")
    h=request.form.get("Slope")
    i=request.form.get("Soil_Type10")
    j=request.form.get("Soil_Type23")
    k=request.form.get("Soil_Type29")
    l=request.form.get("Soil_Type3")
    m=request.form.get("Soil_Type4")
    v=request.form.get("Soil_Type99")
    n=request.form.get("Wilderness_Area1")
    o=request.form.get("Wilderness_Area2")
    p=request.form.get("Wilderness_Area3")
    q=request.form.get("Wilderness_Area4")

    df=pd.DataFrame({"Aspect":[a],"Elevation":[b],"Hillshade_9am":[c],"Hillshade_Noon":[d],"Horizontal_Distance_To_Fire_Points":[e],"Horizontal_Distance_To_Hydrology":[f],"Horizontal_Distance_To_Roadways":[g],"Vertical_Distance_To_Hydrology":[x],
    "Slope":[h],"Soil_Type10":[i],"Soil_Type23":[j],"Soil_Type29":[k],"Soil_Type3":[l],"Soil_Type4":[m],"Wilderness_Area1":[n],"Wilderness_Area2":[o],"Wilderness_Area3":[p]})
    df1=pd.DataFrame(scaler.transform(df),columns=df.columns)
    ans=cluster0.predict(df1)
    if ans==1:
        p="Spruce/Fir"
    elif ans==2:
        p="Lodgepole Pine"
    elif ans==3:
        p="Ponderosa Pine"
    elif ans==4:
        p="Cottonwood/Willow"
    elif ans==5:
        p="Aspen"
    elif ans==6:
        p="Douglas-fir"
    else:
        p="Krummholz"

    return render_template('index.html', prediction_text='Forest Cover Type is {}'.format(p))



if __name__=="__main__":
    app.run(debug=True)