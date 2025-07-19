import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

# Load credentials
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

# Set page config FIRST
st.set_page_config(page_title="Botnet Detector", page_icon="🛡️", layout="wide")

# Login
name, auth_status, username = authenticator.login(location='sidebar', form_name='Login')

if auth_status == False:
    st.error('Username/password is incorrect')
elif auth_status == None:
    st.warning('Please enter your username and password')
elif auth_status:
    authenticator.logout('Logout', 'sidebar')
    st.success(f"Welcome *{name}*! You're logged in.")
    # Proceed with app logic
