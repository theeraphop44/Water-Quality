import streamlit as st
import pandas as pd
#from streamlit_option_menu import option_menu
from menu.Data import run_data_page
from menu.Chart import run_chart_page
from menu.Predict_Model import run_predict_page
from menu.Train_Model import run_train_page
from menu.Homepage_Admin import run_homeadmin_page
from menu.Homepage_User import run_homeuser_page

def sidebar_admin_page():
    logout = st.sidebar.button("Logout")
    if logout == True:
        logout_to_loginpage()
        
    css_style = {
        "icon": {"color": "white"},
        "nav-link": {"--hover-color": "grey"},
        "nav-link-selected": {"background-color": "#FF4C1B"},
    }

    with st.sidebar :
        pages = (
            menu_title = "Main Menu",
            options = ["Homepage","Data","Chart","Train Model","Predict Model"],            
            icons=["house","droplet","droplet","droplet","droplet"],
            styles=css_style
        )

    #Sidebar CSV homepage
    st.sidebar.title("CSV File Upload")
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        # file_name = os.path.basename(uploaded_file.name)
        # st.session_state.file_name = file_name
        df = pd.read_csv(uploaded_file)
        st.session_state["df"] = df
        st.sidebar.markdown('---')
        st.sidebar.success("File uploaded successfully!")
    else :
        st.sidebar.error("Select a page above.")

    if pages == "Homepage":
        run_homeadmin_page() 
    if pages == "Data":
        if uploaded_file is not None:
            run_data_page(df)
            # with st.sidebar :
            #     pages = (
            #         menu_title = None,
            #         options = ["Train Model","Chart"],            
            #         icons=["droplet","droplet"],
            #         styles=css_style
            #     )
        else:
            st.write(" ")
    if pages == "Chart":
        if uploaded_file is not None:
            run_chart_page()
        else:
            st.write(" ")
    if pages == "Predict Model":
        run_predict_page()
    if pages == "Train Model":
        if uploaded_file is not None:
            if "df_replaced" in st.session_state:
                run_train_page()
            else:
                if df.isna().any().any():
                    st.write(" ")
                else :
                    run_train_page()
        else:
            st.write(" ")


def sidebar_user_page():
    logout = st.sidebar.button("Logout")
    if logout == True:
        logout_to_loginpage()
        
    css_style = {
    "icon": {"color": "white"},
    "nav-link": {"--hover-color": "grey"},
    "nav-link-selected": {"background-color": "#FF4C1B"},
    }

    with st.sidebar :
        pages = option_menu(
            menu_title = "Main Menu",
            options = ["Homepage","Predict Model"],            
            icons=["house","droplet"],
            styles=css_style
        )

    if pages == "Homepage":
        run_homeuser_page() 
    if pages == "Predict Model":
        run_predict_page()

# Logout 
def logout_to_loginpage():
    st.session_state["df"] = None
    st.session_state["df_replaced"] = None
    st.session_state.is_logged_in = False
    st.session_state.is_admin = False
    st.session_state.is_user = False

    st.experimental_rerun()
