import streamlit as st
import pandas as pd

# call df and filename
if "df" in st.session_state:
    df = st.session_state["df"]
    df_num = 1
else:
    df_num = 0

if "file_name" in st.session_state:
    file_name = st.session_state['file_name']

# set name page and icon page
st.set_page_config(
    page_title = "Feature and Target",
    page_icon="potable_water:"
)

if df_num == 1:
    # Name file CSV
    st.sidebar.info(file_name)

    # multiselect of Feature
    options = df.columns
    options = options.drop(["Potability"])

    options_feature = st.multiselect(
            'choose feature',
            options
        )  

    # Output of Feature and Target
    col1, col2 = st.columns(2)
    with col1: 
        st.session_state.file_feature = options_feature
        st.title("Features")
        for i in options_feature:
            st.write(i)

    with col2:
        st.title("Target")
        st.write("Potability")
        
    st.markdown('---')


# options_feature = st.session_state['file_feature']
# options_target = st.session_state['file_target']
else :
    st.write("# Pls select file CSV first")