import stqdm
import pickle
from time import sleep
import streamlit as st
from stqdm import stqdm
# call df
if "df" in st.session_state:
    df = st.session_state["df"]
def run_predict_page():

# list model

    name = ['a','b','c','d']

        # Head
    st.write("""<h1>Predict Water Quality</h1>
        <p>Enter these values of the parameters to know if the water quality is suitable to drink or not.</p><hr>
        """, unsafe_allow_html=True)

        # Dropdown
    options = st.selectbox(
            'Select Model',
            (name)
        )

    st.write(options)

    st.markdown('---')

        # Predict model 
    @st.cache_resource
    def load_model():
        with open('assets/model.pkl', 'rb') as f:
            return pickle.load(f)

    col1, col2, col3 = st.columns(3, gap="large")
    with col1:
        ColourTCU = st.number_input(label="Colour (TCU)", min_value=0.0, max_value=1000.0, step=50.0, format="%f",key="test_slider0")
        TurbidityNTU = st.number_input(label="Turbidity (NTU)", min_value=0.0, max_value=1000.0, step=50.0, format="%f",key="test_slider1")
        pH = st.number_input(label="pH", min_value=0.0, max_value=1000.0, step=50.0, format="%f", key="test_slider2")
        ConductivityuS = st.number_input(label="Conductivity (uS/cm)", min_value=0.0, max_value=1000.0, step=50.0,format="%f", key="test_slider3")
        TotalDissolvedSolids = st.number_input(label="Total Dissolved Solids (mg/l)", min_value=0.0, max_value=1000.0,step=50.0, format="%f", key="test_slider4")
        TotalHardness = st.number_input(label="Total Hardness (mg/l as CaCO3)", min_value=0.0, max_value=1000.0,step=50.0, format="%f", key="test_slider5")    
    with col2:
        Aluminium = st.number_input(label="Aluminium (mg/l)", min_value=0.0, max_value=1000.5, step=50.1, format="%f",key="test_slider6")
        Chloride = st.number_input(label="Chloride (mg/l)", min_value=0.0, max_value=1000.0, step=50.0, format="%f",key="test_slider7")
        Iron = st.number_input(label="Iron (mg/l)", min_value=0.0, max_value=1000.5, step=50.1, format="%f",key="test_slider8")
        Sodium = st.number_input(label="Sodium (mg/l)", min_value=0.0, max_value=1000.0, step=50.0, format="%f",key="test_slider9")
        Sulphate = st.number_input(label="Sulphate (mg/l)", min_value=0.0, max_value=1000.0, step=50.0, format="%f",key="test_slider10")
        Zinc = st.number_input(label="Zinc (mg/l)", min_value=0.0, max_value=1000.0, step=50.0, format="%f",key="test_slider11")
    with col3:
        Magnesium = st.number_input(label="Magnesium (mg/l)", min_value=0.0, max_value=1000.0, step=50.0, format="%f",key="test_slider12")
        Calcium = st.number_input(label="Calcium (mg/l)", min_value=0.0, max_value=1000.0, step=50.0, format="%f",key="test_slider13")
        Potassium = st.number_input(label="Potassium (mg/l)", min_value=0.0, max_value=1000.0, step=50.0, format="%f",key="test_slider14")
        Nitrate = st.number_input(label="Nitrate (mg/l)", min_value=0.0, max_value=1000.0, step=50.0, format="%f",key="test_slider15")
        Phosphate = st.number_input(label="Phosphate (mg/l)", min_value=0.0, max_value=1000.2, step=50.1, format="%f",key="test_slider16")
            
        st.write("<br>", unsafe_allow_html=True)
        predict_button = st.button('  Predict Water Quality  ')

    if predict_button:
        model = load_model()
        result = model.predict(df)
        for _ in stqdm(range(50)):
            sleep(0.015)
        if result[0] == 1.0:
            st.error("This Water Quality is Non-Potable")
        else:
            st.success('This Water Quality is Potable')
