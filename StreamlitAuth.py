from tabnanny import check
import jwt
import streamlit as st
from datetime import datetime, timedelta
import extra_streamlit_components as stx
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import Column, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, sessionmaker
from werkzeug.security import check_password_hash
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy import Column, DateTime, Text,String
from datetime import datetime
from sqlalchemy.sql import func
from PIL import Image


"""
    DatabaseConnection Class
    params:
        host: database host
        username: database username
        password: database password
        port: database port
        database: database name
"""
class DatabaseConnection:
    def __init__(self,host,username,password,port,database):

        db_string = 'postgresql://{}:{}@{}:{}/{}'.format(username, password, host, port, database)

        engine = create_engine(db_string, pool_pre_ping=True, echo=False)
        self.db = scoped_session(sessionmaker(
            autocommit=False, autoflush=False, bind=engine))
        self.Base = declarative_base(bind=engine)
    
    """
    User class generator
    returns:
        db: database connection
        UserClass: user class
    """
    def UserClassGenerator(self):

        class User(self.Base):
            __tablename__ = 'users'
            id = Column(UUID(as_uuid=True), primary_key=True)
            name = Column(String(50), nullable=True)
            mail = Column(Text(), nullable=False,unique=True)
            password = Column(Text(), nullable=False)
            create_time = Column(DateTime(timezone=False),
                                default=func.now(), nullable=False)
            update_time = Column(DateTime(timezone=False),
                                default=func.now(), nullable=False)
        return self.db,User



"""
    authenticate class
    params:
        cookie_name: cookie name
        key: secret key
        User: user class
        db: database connection class
        cookie_expiry_days: cookie expiry days

"""
class authenticate:
    def __init__(self,cookie_name,key,User,db,cookie_expiry_days=30):
        self.cookie_name = cookie_name
        self.key = key
        self.cookie_expiry_days = cookie_expiry_days
        self.User = User
        self.db = db

    """
        The encode JWT cookie for passwordless reauthentication.
    """
    def token_encode(self):
        return jwt.encode({'name':st.session_state['name'],
        'exp_date':self.exp_date},self.key,algorithm='HS256')


    """
        The decoded JWT cookie for passwordless reauthentication.
    """
    def token_decode(self):

        return jwt.decode(self.token,self.key,algorithms=['HS256'])
    """
    Calculate the expiration date of the JWT cookie.
    returns:
        exp_date: expiration date of the JWT cookie
    """
    def exp_date(self):

        return (datetime.utcnow() + timedelta(days=self.cookie_expiry_days)).timestamp()


    """
    Check the password of the user.
    returns:
        True: if the password is correct, second parameter is the user name
        False: if the password is incorrect

    """
    def check_pw(self,mail,password):

        UserData = self.db.query(self.User).with_entities(self.User.password,self.User.name).\
        filter(self.User.mail == mail).first()
        if UserData is None:
            return False,None
        else:
            if(check_password_hash(UserData.password,password) == True):
                return True,UserData.name
            else:
                return False,None
    """
        Login page of the app.
    """  
    def Login(self,form_name,location='main'):
        self.location = location
        self.form_name = form_name

        if self.location not in ['main','sidebar']:
            raise ValueError("Location must be one of 'main' or 'sidebar'")

        cookie_manager = stx.CookieManager()

        if 'authentication_status' not in st.session_state:
            st.session_state['authentication_status'] = None
        if 'name' not in st.session_state:
            st.session_state['name'] = None

        if st.session_state['authentication_status'] != True:
            try:
                self.token = cookie_manager.get(self.cookie_name)
                self.token = self.token_decode()

                if self.token['exp_date'] > datetime.utcnow().timestamp():
                    st.session_state['name'] = self.token['name']
                    st.session_state['authentication_status'] = True
                else:
                    st.session_state['authentication_status'] = None
            except:
                st.session_state['authentication_status'] = None

            if st.session_state['authentication_status'] != True:
                if self.location == 'main':
                    col1,col2 = st.columns(2)
                    image = Image.open('images\logo.png')
                    col1.image(image)
                    login_form = col2.form('Login')
                elif self.location == 'sidebar':
                    login_form = st.sidebar.form('Login')


                login_form.subheader(self.form_name)
                Mail = login_form.text_input('E-Mail')
                Password = login_form.text_input('Password',type='password')
                
                if login_form.form_submit_button('Login'):
                    pwstatus, username = self.check_pw(Mail,Password)
                    if (pwstatus == True):
                        st.session_state['name'] = username
                        st.session_state['authentication_status'] = True
                        self.exp_date = self.exp_date()
                        self.token = self.token_encode()

                        cookie_manager.set(self.cookie_name, self.token,
                        expires_at=datetime.now() + timedelta(days=self.cookie_expiry_days))
                    else:
                        st.error("Invalid username or password")
                        st.session_state['authentication_status'] = False

        if st.session_state['authentication_status'] == True:
            if self.location == 'main':
                if st.sidebar.button('Logout'):
                    cookie_manager.delete(self.cookie_name)
                    st.session_state['name'] = None
                    st.session_state['authentication_status'] = None

        elif self.location == 'sidebar':
            if st.sidebar.button('Logout'):
                cookie_manager.delete(self.cookie_name)
                st.session_state['name'] = None
                st.session_state['authentication_status'] = None


        return st.session_state['name'], st.session_state['authentication_status']











