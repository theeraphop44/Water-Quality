import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
    
def replacedata(df):
    df.fillna(method='pad',inplace=True)
    df.fillna(method='bfill',inplace=True)

    #จัดการค่าผิดปกติของผลลัพธ์ที่เป็น 0 (จัดการกับค่าโดยการลบค่าที่ควรเป็น 1 แต่ในdataให้ค่าเป็น 0)
    df = df[~(
        ((df['Hardness'] > 100) & (df['Hardness'] < 300)) &
        ((df['ph'] > 6.5) & (df['ph'] < 8.5)) &
        (df['Sulfate'] < 500) &
        ((df['Conductivity'] > 30) & (df['Conductivity'] < 1500)) &
        (df['Trihalomethanes'] < 80) &
        (df['Turbidity'] < 5) &
        (df['Potability'] == 0)
    )]

    df = df[~(
        (((df['Hardness'] < 100) | (df['Hardness'] > 300)) |
        ((df['ph'] < 6.5) | (df['ph'] > 8.5)) |
        (df['Sulfate'] > 500) |
        ((df['Conductivity'] < 30) | (df['Conductivity'] > 1500)) |
        (df['Trihalomethanes'] > 80) |
        (df['Turbidity'] > 5)) &
        (df['Potability'] == 1)
    )]
    return df
        
def See_data(df_f) :
    st.write("Data from the CSV file:")
    st.write(df_f)
    st.markdown("---")
    st.write("Shape of Data : ",df_f.shape)
    st.markdown("---")
    
def See_nun_lsnull_data(df_f):
    #columns
    col1_nun, col2_nun = st.columns(2) 
    
    #nunique and isnull 

    with col1_nun:
        st.header("Feature")
        nundel_feature = df_f.drop(["Potability"],axis=1)
        ss = nundel_feature.rename(columns={"": 'Feature'})
        st.write(ss.columns)

    with col2_nun:
        st.header("target")
        numdel_target = df_f[['Potability']]
        st.write(numdel_target.columns)

    st.markdown('---')
        
def See_bar_potability_data(df_f):
    # Specify the target folder
    target_folder = "Picture"

    # Create the target folder if it doesn't exist
    os.makedirs(target_folder, exist_ok=True)

    # Count Plot
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df_f, x="Potability", palette="Set1")
    plt.title("Potability Count")
    plt.xlabel("Potability")
    plt.ylabel("Count")

    # Save the count plot image to the target folder
    count_plot_file_path = os.path.join(target_folder, "count_plot.png")
    plt.savefig(count_plot_file_path)

    # Display the image from the target location
    st.image(count_plot_file_path)

    # Optional: Add a centered text below the image
    st.markdown("<div style='text-align: center;'>Potability</div>", unsafe_allow_html=True)
        
    st.markdown('---')

def See_lsnull_data(df_f):
    col1,col2,col3 = st.columns(3)
    with col2:
        st.header("Isnull")
        null = df_f.isnull().sum()
        st.write(null)  
            
def run_data_page(df):
    # Name Title 
    st.markdown('# Water Queality')
    st.markdown('---')
    col4 , col5 , col6 , col7 = st.columns(4)
    st.markdown('---')
    with col4:
        See_Data = st.checkbox("Data",True)
    with col5:
        See_nun_lsnull = st.checkbox("Feature/Target",True)
    with col6:
        See_bar_potability = st.checkbox("Potability Bar",True)
    with col7:
        See_lsnull = st.checkbox("lsnull",True)

    st.write("Replace Data")
    Replace = st.button("Replace")
    st.markdown('---')

    if Replace:
        st.session_state.df_replaced = replacedata(df)
        replacedata(df)
    df_f = st.session_state.df_replaced if "df_replaced" in st.session_state else df
    if See_Data:
        See_data(df_f)
    if See_nun_lsnull:
        See_nun_lsnull_data(df_f)
    if See_bar_potability:
        See_bar_potability_data(df_f)
    if See_lsnull:
        See_lsnull_data(df_f)
