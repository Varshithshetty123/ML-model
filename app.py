import streamlit as st
import pickle
import numpy as np
df=pickle.load(open('dataframe.pkl','rb'))
pipe=pickle.load(open('pipe_model.pkl','rb'))

st.title("Laptop Price Predictor App")
st.write("This app work on the Machine Learning model, created from a small sample of real-world data")
company=st.selectbox("Select the Manufacturer of the laptop",df['Company'].unique(),index=4)
typename=st.selectbox("Select the type of the Laptop",df['TypeName'].unique())
cpu=st.selectbox("Processor Name",df['Cpu'].unique())
ram=st.selectbox("Amount of RAM on the system",[4,8,12,16,24,32,64,128],index=1)
gpu=st.selectbox("Graphics Card",df['Gpu'].unique())
os=st.radio("Operating System",df['OpSys'].unique(),index=2)
weight=st.slider("Weight of the laptop(in kg)",min_value=0.7,max_value=4.8,value=2.2,step=0.1)
touchscreen=st.selectbox("Does the laptop have a touchscreen",[1,0])
ips=st.selectbox("Does the laptop have an IPS display",[1,0])
ppi=st.slider("Pixel Density on the laptop",min_value=90,max_value=350,value=220,step=10)

if st.button("PREDICT PRICE"):
    query=np.array([[company,typename,cpu,ram,gpu,os,weight,touchscreen,ips,ppi]])
    op=pipe.predict(query)
    st.subheader("The laptop with above mentioned configuration will approximatey cost around ₹"
    +str(round(np.exp(op[0]))))
