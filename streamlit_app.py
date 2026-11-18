
import streamlit as st
import requests

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="⭐",
    layout="centered"
)

# Application title
st.title("Product Review Sentiment Analyzer")
st.write("Enter a product review and get instant sentiment prediction!")

# Input box
review = st.text_area(
    "Enter your product review here",
    height=150
)

# Predict button
if st.button("Analyze Sentiment"):

    if not review.strip():
        st.warning("Please enter a review first!")

    else:

        with st.spinner("Analyzing..."):

            try:
                # Call Flask backend
                response = requests.post(
                    "https://amazonesentiment.onrender.com/predict",
                    json={"reviews": review},
                    timeout=30
                )

                # Check API response
                response.raise_for_status()

                # Get JSON result
                result = response.json()

                # Extract sentiment
                sentiment = result['prediction']

                # Display prediction
                if sentiment.lower() == "positive":

                    st.success(f"😊 Sentiment: {sentiment}")

                elif sentiment.lower() == "negative":

                    st.error(f"😞 Sentiment: {sentiment}")

                else:

                    st.warning(f"Unknown sentiment: {sentiment}")

            except requests.exceptions.ConnectionError:

                st.error("Cannot connect to Flask backend. Please run app.py first.")

            except requests.exceptions.RequestException as e:

                st.error(f"API Error: {e}")

            except (KeyError, ValueError):

                st.error("Invalid response received from Flask API.")
