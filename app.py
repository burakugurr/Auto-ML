from random import choice
import streamlit as st
import streamlit_authenticator as stauth
from pages import page2
import StreamlitAuth


# page detail
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(
    page_title="Stok Price Analyis App",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        'Get Help': 'https://github.com/burakugurr',
        'Report a bug': "https://github.com/burakugurr",
        'About': "# Stok Price Analyis App created by Burak Uğur",
    }
)


DbConnection = StreamlitAuth.DatabaseConnection('ec2-52-213-119-221.eu-west-1.compute.amazonaws.com',
                                                'givhjdkqofykpb',
                                                '713e1983a958213f26e62023fb0b5809ceebd86c82d722dfa149d03212cc220c',
                                                '5432', 'd4f45drk5rnlgg')

db, UserClass = DbConnection.UserClassGenerator()

authenticator = StreamlitAuth.authenticate(
'some_cookie_name','some_signature_key',UserClass,db,cookie_expiry_days=5)

name, authentication_status = authenticator.Login('Login','main')
if authentication_status:
    page2.app(db,UserClass,name)

elif authentication_status == False:
    st.error('Username/password is incorrect')
elif authentication_status == None:
    st.warning('Please enter your username and password')
