import streamlit as st
import emoji
import pickle
import pandas as pd
import numpy as np
import os

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
    html_string = f"<div style='text-align: left;'><h1 style='font-family: aptos display; color:purple; font-size:30px;'>Singapore Flats Price Assessor {my_sticker}</h1></div>"
    st.markdown(html_string, unsafe_allow_html=True)
    st.markdown(':rainbow[**_- Resale Price Prediction_**]', unsafe_allow_html=True)

st.write("\n")

st.markdown('## Type your inputs below for prediction')

# Collect inputs
floor_area_sqm = st.number_input("Enter floor area (sqm) here", min_value=1.0)
lease_commence_date = st.number_input("Enter lease commenced year here", min_value=1970, max_value=pd.Timestamp.now().year)
remaining_lease = st.text_input("Enter remaining lease duration here (e.g., 60 years 05 months)")
flat_model = st.selectbox("Select a model of flat from the dropdown list:", ['---','2-room', '3Gen', 'Adjoined flat', 'Apartment', 'DBSS', 'Improved', 'Improved-Maisonette', 'Maisonette', 'Model A', 'Model A2', 'Model A-Maisonette', 'Multi Generation', 'New Generation', 'Premium Apartment', 'Premium Apartment Loft', 'Premium Maisonette', 'Simplified', 'Standard', 'Terrace', 'Type S1', 'Type S2'])
flat_type = st.selectbox("Select a type of flat from the dropdown menu:", ['---','1 ROOM', '2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE', 'MULTI-GENERATION'])
storey_range = st.selectbox("Select the storey level:", ['---', '04 TO 06', '07 TO 09',	'10 TO 12', '01 TO 03',	'13 TO 15', '16 TO 18',	'19 TO 21', '22 TO 24', '01 TO 05','06 TO 10', '25 TO 27', '11 TO 15', '28 TO 30', '31 TO 33', '34 TO 36', '37 TO 39', '16 TO 20', '40 TO 42', '21 TO 25', 
                                                         '43 TO 45', '46 TO 48', '26 TO 30', '49 TO 51', '36 TO 40', '31 TO 35'])
town = st.selectbox("Choose a town from the list:", ['---','ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH', 'BUKIT PANJANG', 'BUKIT TIMAH', 'CENTRAL AREA', 'CHOA CHU KANG', 'CLEMENTI', 'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST', 'KALLANG/WHAMPOA', 'MARINE PARADE', 'PASIR RIS', 'PUNGGOL', 'QUEENSTOWN', 'SEMBAWANG', 'SENGKANG', 'SERANGOON', 'TAMPINES', 'TOA PAYOH', 'WOODLANDS', 'YISHUN'])

clicked1 = st.button("Predict Resale Price")

# Load the model and encoders
# Define the relative path to the model
model_path = os.path.join('Capstone 5', 'best_model.pkl')
town_encoder_path = os.path.join('Capstone 5', 'town_encoder3.pkl')
flat_type_encoder_path = os.path.join('Capstone 5', 'flat_type_encoder3.pkl')
storey_range_encoder_path = os.path.join('Capstone 5', 'storey_range_encoder3.pkl')
flat_model_encoder_path = os.path.join('Capstone 5', 'flat_model_encoder3.pkl')

# Load the model and encoders using the relative path
with open(model_path, 'rb') as file_a:
    dtr = pickle.load(file_a)
with open(town_encoder_path, 'rb') as file_b:
    town_encoder = pickle.load(file_b)
with open(flat_type_encoder_path, 'rb') as file_c:
    flat_type_encoder = pickle.load(file_c)
with open(storey_range_encoder_path, 'rb') as file_d:
    storey_range_encoder = pickle.load(file_d)
with open(flat_model_encoder_path, 'rb') as file_e:
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

        X_trail = input_df

        prediction = dtr.predict(X_trail)
        st.write(f"Predicted Price(SGD): {prediction[0]:.2f}")
    except ValueError as e:
        st.error(f"Error: {str(e)}")
    except KeyError as e:
        st.error(f"Unseen category found - {str(e)}")
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")

    st.write("\n")
    st.write("\n")
    
    st.write("""Disclaimer: 
             \n Please note that the predicted resale prices provided by the application are based on historical data and a machine learning model. 
             \n They should be used as estimates and not as absolute values for real estate transactions. 
             \n It's advisable to consult with real estate professionals for accurate and up-to-date pricing information. 
             \n Thank you for using our Resale Price Assessor Web Application!""")
st.write("\n")
