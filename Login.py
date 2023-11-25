import streamlit as st
from menu.sidebar import sidebar_admin_page, sidebar_user_page

def main():

    # ในที่นี้เราจะใช้ Session State เพื่อเก็บข้อมูลสถานะ
    session_state = st.session_state

    if not hasattr(session_state, "is_logged_in"):
        session_state.is_logged_in = False
    if "is_admin" not in st.session_state:
        st.session_state.is_admin = False
    if "is_user" not in st.session_state:
        st.session_state.is_user = False

    if not session_state.is_logged_in:
        login()
    else:
        # เมื่อ login สำเร็จ, เลือกแสดงหน้าต่างๆ ตามต้องการ
        if st.session_state.is_admin:
            sidebar_admin_page()
        elif st.session_state.is_user:
            sidebar_user_page()

  
def login():
    x = "admin"
    y = "123"

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    login_button = st.button("Login")

    if login_button:
        # ตรวจสอบการ login ที่นี่
        if username == x and password == y:
            st.success("Login Successful as Admin")
            st.session_state.is_logged_in = True
            st.session_state.is_admin = True
            main()
        elif username == "user" and password == "123":
            st.success("Login Successful as User")
            st.session_state.is_logged_in = True
            st.session_state.is_user = True
            main()
            
        else:
            st.error("Username or Password is not correct")


if __name__ == "__main__":
    main()