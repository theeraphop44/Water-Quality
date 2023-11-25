import streamlit as st
from stqdm import stqdm
from time import sleep

# Check if df is already in session state
if "df" not in st.session_state:
    st.session_state["df"] = None

def run_homeuser_page():
    
    #Homepage
    st.markdown("# Water Quality")
    cover_image_html = """
    <div style="text-align: center; padding: 0px;">
        <img src="https://www.vivaqua.be/content/uploads/2021/02/iStock-1161576130-1400x933.jpg" alt="Cover Image" style="max-width: 100%;">
    </div>
    """
    # Render the homepage image using st.markdown
    st.markdown(cover_image_html, unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown("""
        Drinking water is a vital resource for human health and well-being. It is essential for staying hydrated, maintaining bodily functions, and preventing dehydration. Access to clean and safe drinking water is a basic human right, yet many people around the world do not have access to it. Contaminated water can cause waterborne diseases, such as cholera and dysentery, which can be deadly, particularly for children and those with weakened immune systems. Ensuring that everyone has access to clean drinking water is crucial for public health and a fundamental responsibility of governments worldwide.
                """)
    st.caption('Project _nearing_ :red[deadlie]:  🥹:') 
    