import streamlit as st
import emoji
import pickle
import pandas as pd
import numpy as np

st.markdown("""
    <style>
        /* Adjust padding in columns */
        .css-1d391kg {
            padding: 4000px;
        }

        /* Adjust the padding and background color of the block container */
        .block-container {
            padding: 5px;
            margin: 2px;  /* Add a semicolon here */
            background-color: #90FF65;
        }

        /* Adjust body styling */
        body {
            line-height: 1.3; 
        }

        /* Change the background color of the main container */
        .main {
            background-color: #90FF65; /* Specify any background color here */
        }
         
        /* Change the background color of the sidebar */
        [data-testid="stSidebar"] {
            background-color: #90FF65; /* Light blue color */
        }
            
        /* Set background color of the chart */
            div[data-testid="stPlotlyChart"] div.plotly-graph-div {
            background-color: transparent !important;
            }
    </style>
""", unsafe_allow_html=True)

top_bar = st.container()
with top_bar:
    my_sticker = emoji.emojize (':cityscape:')
    st.write(my_sticker)
    html_string = f"<div style='text-align: left;'><h1 style='font-family: aptos display; color:purple; font-size:30px;'>Singapore Flats Resale Price Predictor {my_sticker}</h1></div>"
    st.markdown(html_string, unsafe_allow_html=True)
    st.markdown(':rainbow[**_- Resale Price Prediction_**]', unsafe_allow_html=True)

st.write("\n")


import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.markdown('## Type your inputs below for prediction')

# Collect inputs
floor_area_sqm = st.number_input("Enter floor area (sqm) here", min_value=1.0)
lease_commence_date = st.number_input("Enter lease commenced year here", min_value=1970, max_value=pd.Timestamp.now().year)
remaining_lease = st.text_input("Enter remaining lease duration here (e.g., 60 years 05 months)")
flat_model = st.selectbox("Select a model of flat from the dropdown list:", ['---','2-room', '3Gen', 'Adjoined flat', 'Apartment', 'DBSS', 'Improved', 'Improved-Maisonette', 'Maisonette', 'Model A', 'Model A2', 'Model A-Maisonette', 'Multi Generation', 'New Generation', 'Premium Apartment', 'Premium Apartment Loft', 'Premium Maisonette', 'Simplified', 'Standard', 'Terrace', 'Type S1', 'Type S2'])
flat_type = st.selectbox("Select a type of flat from the dropdown menu:", ['---','1 ROOM', '2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE', 'MULTI-GENERATION'])
storey_range = st.text_input("Enter storey range here (e.g., 13 TO 15)")
town = st.selectbox("Choose a town from the list:", ['---','ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH', 'BUKIT PANJANG', 'BUKIT TIMAH', 'CENTRAL AREA', 'CHOA CHU KANG', 'CLEMENTI', 'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST', 'KALLANG/WHAMPOA', 'MARINE PARADE', 'PASIR RIS', 'PUNGGOL', 'QUEENSTOWN', 'SEMBAWANG', 'SENGKANG', 'SERANGOON', 'TAMPINES', 'TOA PAYOH', 'WOODLANDS', 'YISHUN'])

clicked1 = st.button("Predict Resale Price")

# Load the model and encoders
model_path = 'C:/Users/my pc/Cap 5/'
with open(model_path + 'rfr2_model.pkl', 'rb') as file_a:
    rfr = pickle.load(file_a)
with open(model_path + 'town_encoder2.pkl', 'rb') as file_b:
    town_encoder = pickle.load(file_b)
with open(model_path + 'flat_type_encoder2.pkl', 'rb') as file_c:
    flat_type_encoder = pickle.load(file_c)
with open(model_path + 'storey_range_encoder2.pkl', 'rb') as file_d:
    storey_range_encoder = pickle.load(file_d)
with open(model_path + 'flat_model_encoder2.pkl', 'rb') as file_e:
    flat_model_encoder = pickle.load(file_e)

current_year = pd.Timestamp.now().year

if remaining_lease:
    try:
        years_part, months_part = remaining_lease.split('years')
        years = int(years_part.strip())
        months = int(months_part.split('months')[0].strip())
        remaining_lease_months = years * 12 + months
    except Exception as e:
        st.error(f"Invalid format for remaining lease. Error: {str(e)}")
        st.stop()


def num_rooms(flat_type):
    mapping = {
        '1 ROOM': 1,
        '2 ROOM': 2,
        '3 ROOM': 3,
        '4 ROOM': 4,
        '5 ROOM': 4.5,
        'EXECUTIVE': 6,
        'MULTI GENERATION': 5
    }
    return mapping.get(flat_type, 0)

number_rooms = num_rooms(flat_type)

if clicked1:
    try:
        input_data = {
            'floor_area_sqm': [floor_area_sqm],
            'lease_commence_date': [lease_commence_date],
            'age_of_flat': [current_year - lease_commence_date],
            'remaining_lease_months': [remaining_lease_months],
            'flat_model': [flat_model],
            'num_rooms': [number_rooms],
            'flat_type': [flat_type],
            'storey_range': [storey_range],
            'town': [town]
        }

        input_df = pd.DataFrame(input_data)

        input_df['town_encoded'] = town_encoder.transform(input_df['town'])
        input_df['flat_type_encoded'] = flat_type_encoder.transform(input_df['flat_type'])
        input_df['storey_range_encoded'] = storey_range_encoder.transform([input_df['storey_range'][0]])
        input_df['flat_model_encoded'] = flat_model_encoder.transform(input_df['flat_model'])

        input_df = input_df.drop(columns=['town', 'flat_type', 'storey_range', 'flat_model'])

        X_trail = input_df.to_numpy()

        prediction = rfr.predict(X_trail)
        st.write(f"Predicted Price(SGD): {prediction[0]}")
    except ValueError as e:
        st.error(f"Error: {str(e)}")
    except KeyError as e:
        st.error(f"Unseen category found - {str(e)}")
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")



st.write("\n")