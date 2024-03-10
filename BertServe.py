import streamlit as st
import requests

# Function to send a request to your model's API
def get_prediction(description):
    # Replace the URL with the endpoint where your model is served
    url = 'http://localhost:8000/predict'
    # Assuming the API expects a JSON with a field 'description' containing the text
    data = {'description': description}
    # Send a POST request to the model API
    response = requests.post(url, json=data)
    # Assuming the response contains a field 'label' with the predicted label
    return response.json()['label']

# Streamlit app
def main():
    st.title('BERT Model Prediction')
    
    # Text input from user
    user_input = st.text_area("Enter description text", "")
    
    # Button to make prediction
    if st.button('Predict'):
        if user_input:  # Check if the input is not empty
            # Get the prediction
            label = get_prediction(user_input)
            # Display the result
            st.write(f'Predicted Label: {label}')
        else:
            st.write('Please enter a description text.')

if __name__ == "__main__":
    main()
