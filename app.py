from Multipage import MultiPage
import streamlit as st
from PIL import Image
from pages import page2,page1

# Create an instance of the app 
app = MultiPage()


st.title("Welcome to Stock App")
st.subheader("Stock App is a web app that can help you to predict stock prices and anomaly detection")
image = Image.open(r'.\temp\mainpage.jpg')
st.image(image, caption='Photo by Maxim Hopman on Unsplash',use_column_width='auto')

app.add_page("Login", page1.app)
app.add_page("Home Page", page2.app)



app.run()