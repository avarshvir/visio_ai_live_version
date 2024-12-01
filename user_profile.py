import streamlit as st
import mysql.connector
from auth import fetch_history

def profile_page():
    st.title("👤 User Profile")
    
    # Check if user is logged in
    if not st.session_state.get('logged_in'):
        st.error("Please log in to view your profile")
        return
    
    try:
        # Establish database connection
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="user_db"
        )
        cursor = conn.cursor(dictionary=True)
        
        # Fetch user details
        query = "SELECT * FROM users WHERE email = %s"
        cursor.execute(query, (st.session_state.user_email,))
        user_details = cursor.fetchone()
        
        if user_details:
            # Profile Information Section
            st.header("📋 Personal Information")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Name:** {user_details['name']}")
                st.write(f"**Email:** {user_details['email']}")
                st.write(f"**Date of Birth:** {user_details['dob']}")
            
            with col2:
                st.write(f"**Address:** {user_details['address']}")
            
            # User History Section
            st.header("📜 Work History")
            history = fetch_history(st.session_state.user_email)
            
            if history:
                #st.write(f"**Last Worked File:** {history[0]}")
                #st.write(f"**Last Activity Date:** {history[1]}")
                #t.write(f"**Last Worked File:** {history.get('last_worked_file', 'N/A')}")
                #st.write(f"**Last Activity Date:** {history.get('last_activity_date', 'N/A')}")
                st.subheader("Recent Files Worked On")
                for entry in history:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**File:** {entry.get('last_worked_file', 'N/A')}")
                    with col2:
                        st.write(f"**Date:** {entry.get('last_activity_date', 'N/A')}")
                    st.divider()
            else:
                st.info("No recent work history found.")
            
            # Profile Update Section
            st.header("🔧 Update Profile")
            with st.form("profile_update"):
                new_name = st.text_input("Update Name", value=user_details['name'])
                new_address = st.text_area("Update Address", value=user_details['address'])
                
                if st.form_submit_button("Update Profile"):
                    update_query = "UPDATE users SET name = %s, address = %s WHERE email = %s"
                    cursor.execute(update_query, (new_name, new_address, st.session_state.user_email))
                    conn.commit()
                    st.success("Profile updated successfully!")
                    st.experimental_rerun()
        
        else:
            st.error("User details not found.")
    
    except mysql.connector.Error as err:
        st.error(f"Database error: {err}")
    
    finally:
        # Close database connections
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

# This function can be called from home.py when profile button is clicked
def show_profile():
    profile_page()