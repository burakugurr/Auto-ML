
from sqlite3 import IntegrityError
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import Column, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, sessionmaker
from werkzeug.security import check_password_hash, generate_password_hash
from sqlalchemy import ForeignKey
from sqlalchemy.dialects.postgresql import UUID,ARRAY
from sqlalchemy import Column, DateTime, Text, JSON, Boolean,Float,Integer,Time,Numeric,String
import datetime
from sqlalchemy.sql import func
import uuid
import time



global user_data
# define Postgresql connection
db_string = 'postgresql://givhjdkqofykpb:713e1983a958213f26e62023fb0b5809ceebd86c82d722dfa149d03212cc220c@ec2-52-213-119-221.eu-west-1.compute.amazonaws.com:5432/d4f45drk5rnlgg'


engine = create_engine(db_string, pool_pre_ping=True, echo=False)
db = scoped_session(sessionmaker(
    autocommit=False, autoflush=False, bind=engine))
Base = declarative_base(bind=engine)


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


class User(Base):
    __tablename__ = 'users'
    id = Column(UUID(as_uuid=True), primary_key=True)
    name = Column(String(50), nullable=True)
    mail = Column(Text(), nullable=False,unique=True)
    password = Column(Text(), nullable=False)
    create_time = Column(DateTime(timezone=False),
                         default=func.now(), nullable=False)
    update_time = Column(DateTime(timezone=False),
                         default=func.now(), nullable=False)

def create_user(mail,password,name):
    useruuid = uuid.uuid4()
    hashed_password = generate_password_hash(password+str(useruuid), method='sha256')
    new_user = User(id=useruuid,
                    name=name,
                    mail=mail,
                    password=hashed_password,
                    create_time=datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
                    update_time=datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
    try:
        db.add(new_user)
        db.commit()
    except:
        return False
    return True


def check_password(mail,password):
    user = db.query(User).filter_by(mail=mail).first()
    if user is not None:
        check_password_hash(user.password,password)
        return True
    return False


def get_user_id(email,table):
    user = db.query(table).filter_by(mail=email).first()
    if user is not None:
        user_data = {}
        user_data['name'] = user.name

        return user_data
    return False

def update_user(user_id,table,data):
    user = db.query(table).filter_by(user_id=user_id).first()
    if( user is None):
        return False
    user.user_name = data['name']
    user.user_mail = data['mail']
    user.user_password = data['password']
    user.update_time = datetime.now().strftime("%Y/%m/%d %H:%M:%S")

    db.commit()
    return {'id':user.user_name,'success':True}
    

def delete_user(id,table):
    user = db.query(table).filter_by(id=id).first()
    if user is not None:
        db.session.delete(user)
        db.session.commit()
        return True
    return False






# App starts here


def app():
    text_input_container = st.empty()

    status = text_input_container.select_slider(
        'Status',
        options=['Sign Up', 'Login'])

    if(status == "Sign Up"):
        with st.form("my_form"):
            mail = st.text_input("Email",help="Enter your email")
            name = st.text_input("Name",help="Enter your name")
            password = st.text_input("Passwords",type='password',help="Enter your password")
            submitted = st.form_submit_button("Submit")
            
            if(submitted):
                with text_input_container.spinner("Loading.. Please wait"):
                    createStatus = create_user(mail,password,name)
                    if createStatus == True:
                        text_input_container.success("User created successfully")
                        text_input_container.balloons()
                        time.sleep(2)
                        text_input_container.empty()
                    else:
                        text_input_container.error("User already exists")
                        

    else:
        text_input_container.write("Login to your account")
        with st.form("my_form"):
            mail = st.text_input("Email",help="Enter your e-mail")
            password = st.text_input("Password",type='password',help="Enter your password")
            submitted = st.form_submit_button("Submit")              
              
            if submitted: 
                isuser = check_password(mail,password)
                if isuser == True:
                    
                    user_data = get_user_id(mail,User)
                    text_input_container.success("Welcome "+ user_data['name'])
                
                    
                else:
                    text_input_container.error("Invalid email or password")
                    

