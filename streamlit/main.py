import streamlit as st
import requests
import json
import pandas as pd

st.header("CIGIN")
form = st.form("prediction_form")
solute = form.text_input("Enter Solute")
solvent = form.text_input("Enter Solvent")
submit_button = form.form_submit_button("Predict")

if submit_button == True:
    response = requests.get(f"https://d4-cigin.herokuapp.com/prediction?solute={solute}&solvent={solvent}")
    json = json.loads(response.text)
    st.metric("SOLVATION FREE ENERGY",json["prediction"]["solvation"])
    df = pd.DataFrame(json["prediction"]["interaction_map"])
    st.write("Interaction Map",df)
    