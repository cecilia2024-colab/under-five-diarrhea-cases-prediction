import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

df = pd.read_csv('diarrhea_data.csv')
df['periodname'] = pd.to_datetime(df['periodname'], format='%b-%y')

# Load the trained model
with open('diarrhea_prediction_model_year_month.pkl', 'rb') as file:
    model = pickle.load(file)

st.image("engage_jooust_branding.png",caption="Engage brands")
st.markdown("[ENGAGE Program](https://engage.uonbi.ac.ke)")

# Title of the app
st.title('Diarrhea Cases Prediction for Children Under Five')

# Instructions
st.write('Enter the values for the features to predict the number of diarrhea cases.')

# Input fields for the features
# Assuming 'month' is an integer representing the month (e.g., 1 for January, 2 for February, etc.)
month = st.slider('Month (1-12)', min_value=0, max_value=12, value=1)

# Assuming 'county' is a categorical feature, provide a dropdown or numerical code if appropriate
# Example: 'county' could be a numerical code; adjust according to your dataset
#county = st.number_input('County code', min_value=1, max_value=47, value=1)
county = st.selectbox('County code', df['organisationunitname'].unique())

current_year = datetime.now().year
years = list(range(2018, 2023))
year = st.selectbox('Year', years)
year_enc = {2018: 0, 2019: 1, 2020: 2, 2021: 3, 2022: 4}
# Convert inputs to a format the model can accept
#features = np.array([[year_enc[int(year)],month, county]])
features = np.array([[year_enc[int(year)],month]])

# Prediction
if st.button('Predict'):
    prediction = model.predict(features)
    st.write(f'Predicted diarrhea cases: {int(prediction[0])}')

# For visualization (if you have time-series data for trend analysis)
if st.checkbox('Show Trend Visualization'):
    # Replace with actual trend data if available
    # Group the data by county and periodname (month)
    grouped = df.groupby(['organisationunitname', 'periodname'])['Total under five Diarrhoea cases reported'].sum().reset_index()

    # Choose the county you want to plot
    county_to_plot =  county # Replace with the desired county name

    # Filter the data for the specific county
    subset = grouped[grouped['organisationunitname'] == county_to_plot]

    # Plot the data
    fig, ax = plt.subplots()
    fig.set_size_inches(12,8)

    #ax.figure(figsize=(10, 6))
    ax.plot(subset['periodname'], subset['Total under five Diarrhoea cases reported'], label=county_to_plot)

    ax.set_xlabel('Month')
    ax.set_ylabel('Total Cases')
    ax.set_title(f'Diarrhoea Cases Over Time in {county_to_plot}')  # Dynamic title
    ax.legend()
    #ax.set_grid(True)
    
    ax.set_xticklabels(subset['periodname'], rotation=90) # Rotate x-axis labels for better readability
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    #ax.tight_layout() # Adjust layout to prevent labels from overlapping
    #plt.show()
    
    st.pyplot(fig)
