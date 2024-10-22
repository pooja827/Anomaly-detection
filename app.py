import streamlit as st
import sqlite3
import pandas as pd
import time
import numpy as np
from datetime import datetime
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import os

# Create a connection to the SQLite database
def create_connection():
    conn = sqlite3.connect('sensor_data.db')
    return conn

# Create the SensorReadings table if it doesn't exist
def create_table():
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS SensorReadings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        ph REAL,
        turbidity REAL,
        hardness REAL,
        solids REAL,
        conductivity REAL,
        organic_carbon REAL
    )
    ''')
    conn.commit()
    conn.close()

# Function to generate random sensor readings with normalization
def generate_sensor_data(previous_data, failure_scenario=None):
    ph_bounds = (6.5, 8.5)
    turbidity_bounds = (0, 10)  # NTU
    hardness_bounds = (50, 300)  # mg/L
    solids_bounds = (100, 1000)  # mg/L
    conductivity_bounds = (100, 500)  # µS/cm
    organic_carbon_bounds = (1, 10)  # mg/L

    if failure_scenario == 'immediate':
        turbidity = np.random.uniform(11, 15)  # Non-potable condition
        ph = np.random.uniform(6.0, 6.5)
        hardness = np.random.uniform(400, 500)
        solids = np.random.uniform(1000, 1500)
        conductivity = np.random.uniform(600, 1000)
        organic_carbon = np.random.uniform(10, 15)
    elif failure_scenario == 'gradual':
        turbidity = np.clip(previous_data['turbidity'] + np.random.normal(0, 1), 0, 10)  # Increase turbidity
        ph = np.clip(previous_data['ph'] - 0.1, 6.0, 8.5)  # Gradual decrease in pH
        hardness = np.clip(previous_data['hardness'] + np.random.normal(0, 1), 50, 300)
        solids = np.clip(previous_data['solids'] + np.random.normal(0, 5), 100, 1000)
        conductivity = np.clip(previous_data['conductivity'] + np.random.normal(0, 5), 100, 500)
        organic_carbon = np.clip(previous_data['organic_carbon'] + np.random.normal(0, 0.1), 1, 10)
    else:
        ph = np.clip(previous_data['ph'] + np.random.normal(0, 0.1), *ph_bounds)
        turbidity = np.clip(previous_data['turbidity'] + np.random.normal(0, 0.5), *turbidity_bounds)
        hardness = np.clip(previous_data['hardness'] + np.random.normal(0, 5), *hardness_bounds)
        solids = np.clip(previous_data['solids'] + np.random.normal(0, 10), *solids_bounds)
        conductivity = np.clip(previous_data['conductivity'] + np.random.normal(0, 1), *conductivity_bounds)
        organic_carbon = np.clip(previous_data['organic_carbon'] + np.random.normal(0, 0.1), *organic_carbon_bounds)

    return ph, turbidity, hardness, solids, conductivity, organic_carbon

# Insert sensor data into the database
def insert_sensor_data(failure_scenario=None):
    conn = create_connection()
    cursor = conn.cursor()

    previous_data = load_data()
    if previous_data.empty:
        previous_data = {
            'ph': 7,
            'turbidity': 5,
            'hardness': 150,
            'solids': 500,
            'conductivity': 200,
            'organic_carbon': 5
        }
    else:
        previous_data = previous_data.iloc[0]

    ph, turbidity, hardness, solids, conductivity, organic_carbon = generate_sensor_data(previous_data, failure_scenario)

    cursor.execute('''
    INSERT INTO SensorReadings (timestamp, ph, turbidity, hardness, solids, conductivity, organic_carbon)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (datetime.now(), ph, turbidity, hardness, solids, conductivity, organic_carbon))

    conn.commit()
    conn.close()

# Load data from the SQLite database
def load_data():
    conn = create_connection()
    df = pd.read_sql_query("SELECT * FROM SensorReadings ORDER BY timestamp DESC", conn)
    conn.close()
    return df

# Load pre-trained LSTM model with error handling
def load_lstm_model():
    model_path = 'water_potability_lstm_model.h5'

    # Check if the model file exists
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found. Please ensure the file is in the correct directory.")
        return None

    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Preprocess the data for LSTM
def preprocess_data(data):
    # Select the relevant features
    data = data[['ph', 'turbidity', 'hardness', 'solids', 'conductivity', 'organic_carbon']]

    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Create a sequence with a look-back period of 10
    sequence_length = 10
    sequences = []
    if len(data_scaled) >= sequence_length:
        sequences.append(data_scaled[-sequence_length:])  # Use the last 10 readings

    return np.array(sequences)

# Predict potability using the LSTM model
def predict_potability(data, model):
    if data.shape[0] == 0 or model is None:
        return None  # No data or no model available yet
    sequence = preprocess_data(data)
    prediction = model.predict(sequence)
    return (prediction > 0.5).astype(int)[0][0]

# Set up the Streamlit app
st.title("Real-Time Sensor Data Dashboard with LSTM Predictions")

# Create the database table if it doesn't already exist
create_table()

# Initialize session state for data generation
if 'running' not in st.session_state:
    st.session_state.running = False
if 'failure_counter' not in st.session_state:
    st.session_state.failure_counter = 0

# Start or stop data generation based on the button clicked
if st.button("Start Generating Sensor Data"):
    st.session_state.running = True

if st.button("Stop Generating Sensor Data"):
    st.session_state.running = False

# Load the LSTM model
lstm_model = load_lstm_model()

# Initialize session state for prediction history
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Continuous data generation and display
placeholder = st.empty()  # For continuous updates

if st.session_state.running:
    while True:
        # Check for failure scenarios
        if st.session_state.failure_counter < 10:
            failure_scenario = None  # No failure
        elif st.session_state.failure_counter < 20:
            failure_scenario = 'gradual'  # Gradual failure scenario
        else:
            failure_scenario = 'immediate'  # Immediate failure scenario

        # Generate and insert new sensor data
        insert_sensor_data(failure_scenario)

        # Load the latest data from the database
        data = load_data()

        # Display the data in the Streamlit app
        with placeholder.container():
            st.subheader("Sensor Readings (Latest 10 Entries)")
            st.dataframe(data.head(10))  # Show the latest 10 entries

            if not data.empty:
                st.subheader("Latest Sensor Reading")
                st.write(data.iloc[0])  # Show the most recent reading

                # Predict potability based on real-time data
                prediction = predict_potability(data, lstm_model)
                if prediction is not None:
                    st.subheader(f"Potability Prediction: {'Potable' if prediction == 0 else 'Not Potable'}")

                    # Handle prediction and failure scenarios
                    if prediction == 1:  # Not potable
                        st.session_state.failure_counter += 1
                        st.warning("Alert: Water is NOT potable!")
                        st.error("Notification for device 0001 is delivered to mkservices12@gmail.com")

                        # Display features responsible for failure
                        st.subheader("Features Responsible for Failure:")
                        st.write(f"Turbidity: {data.iloc[0]['turbidity']:.2f}")
                        st.write(f"pH: {data.iloc[0]['ph']:.2f}")

                    else:  # Potable
                        if st.session_state.failure_counter > 0:
                            st.session_state.failure_counter = 0  # Reset counter
                        st.success("Water is potable.")

                    # Add the latest prediction to the prediction history
                    st.session_state.prediction_history.append({
                        'timestamp': data.iloc[0]['timestamp'],
                        'potability': prediction
                    })

                # Convert prediction history to DataFrame for plotting
                prediction_df = pd.DataFrame(st.session_state.prediction_history)

                if not prediction_df.empty:
                    st.subheader("Potability Prediction History")
                    st.line_chart(prediction_df.set_index('timestamp')['potability'].astype(int), use_container_width=True)

                # Customizing the chart background
                st.markdown(
                    """
                    <style>
                    .streamlit-expanderHeader {
                        color: #FFFFFF;
                    }
                    .stLineChart {
                        background-color: black;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )

                # Plot graphs for each feature
                st.subheader("Real-Time Graphs of Sensor Readings")
                st.line_chart(data[['timestamp', 'ph']].set_index('timestamp'), use_container_width=True)
                st.line_chart(data[['timestamp', 'turbidity']].set_index('timestamp'), use_container_width=True)
                st.line_chart(data[['timestamp', 'hardness']].set_index('timestamp'), use_container_width=True)
                st.line_chart(data[['timestamp', 'solids']].set_index('timestamp'), use_container_width=True)
                st.line_chart(data[['timestamp', 'conductivity']].set_index('timestamp'), use_container_width=True)
                st.line_chart(data[['timestamp', 'organic_carbon']].set_index('timestamp'), use_container_width=True)

        time.sleep(1)  # Wait 1 second before updating
else:
    st.info("Data generation is currently stopped.")
