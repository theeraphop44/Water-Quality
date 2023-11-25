import streamlit as st
from menu.sidebar import sidebar_admin_page, sidebar_user_page
import sqlite3


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
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    login_button = st.button("Login")

    if login_button:
        # ทำการเชื่อมต่อกับฐานข้อมูล
        conn = sqlite3.connect("./DB/data.db")
        cursor = conn.cursor()

        # ทำการ query หรือตรวจสอบข้อมูลในฐานข้อมูล
        query = f"SELECT * FROM user WHERE username='{username}' AND password='{password}'"
        cursor.execute(query)

        # ดึงข้อมูลที่ได้จาก query
        result = cursor.fetchone()

        # ตรวจสอบว่า username และ password ถูกต้องหรือไม่
        if result:
            st.success("Login Successful!")

            # ตรวจสอบว่าเป็น user หรือ admin
            if result[3] == 'admin':
                st.info("Logged in as admin")
                st.session_state.is_logged_in = True
                st.session_state.is_admin = True
                main()
            else:
                st.info("Logged in as user")
                st.session_state.is_logged_in = True
                st.session_state.is_user = True
                main()
        else:
            st.error("Username or Password is not correct")

        # ปิดการเชื่อมต่อกับฐานข้อมูล
        conn.close()

if __name__ == "__main__":
    main()