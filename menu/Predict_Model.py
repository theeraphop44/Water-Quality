import stqdm
import pickle
from time import sleep
import streamlit as st
from stqdm import stqdm
import numpy as np
from joblib import load
import sqlite3
import pandas as pd
# call df
if "df" in st.session_state:
    df = st.session_state["df"]
def run_predict_page():

# list model

    

        # Head
    st.write("""<h1>Predict Water Quality</h1>
        <p>Enter these values of the parameters to know if the water quality is suitable to drink or not.</p><hr>
        """, unsafe_allow_html=True)


        # Predict model 
   
    
    conn = sqlite3.connect("./DB/data.db")  # แทน your_database.db ด้วยชื่อฐานข้อมูลของคุณ
    query = "SELECT * FROM pathmodels;"  # แทน your_table ด้วยชื่อตารางที่มีข้อมูล name
    data = pd.read_sql(query, conn)
    conn.close()
    
    selected_name = st.selectbox("เลือกชื่อ:", data['name'])
    selected_path = data.loc[data['name'] == selected_name, 'path'].iloc[0]
    testaccuracy = data.loc[data['name'] == selected_name, 'testaccuracy'].iloc[0]
    trainaccuracy = data.loc[data['name'] == selected_name, 'trainaccuracy'].iloc[0]
    st.write("คุณเลือก: ", selected_name)
    st.write("path: ", selected_path)
    st.success(trainaccuracy)
    st.success(testaccuracy)
    
    
    loaded_model = load(selected_path)
    ph = st.number_input('Enter PH:', min_value=0.0, max_value=14.0, value=7.0)
    hardness = st.number_input('Enter Hardness:', min_value=47.0, value=323.0)
    solids = st.number_input('Enter Solids:', min_value=323.0, value=61227.0)
    chloramines = st.number_input('Enter Chloramines:', min_value=1.0, value=13.0)
    sulfate = st.number_input('Enter Sulfate:', min_value=129.0, value=481.0)
    conductivity = st.number_input('Enter Conductivity:', min_value=181.0, value=753.0)
    organic_carbon = st.number_input('Enter Organic Carbon:', min_value=2.0, value=28.0)
    trihalomethanes = st.number_input('Enter Trihalomethanes:', min_value=0.0, value=124.0)
    turbidity = st.number_input('Enter Turbidity:', min_value=1.0, value=6.0)
    
    predict_button = st.button('  Predict Water Quality  ')
    if predict_button:
        # ให้โมเดลทำนาย
        input_data = np.array([[ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity]])
        prediction = loaded_model.predict(input_data)

        for _ in stqdm(range(50)):
            sleep(0.015)
        if prediction[0] == 0:
            st.error("This Water Quality is Non-Potable")
        else:
            st.success('This Water Quality is Potable')
