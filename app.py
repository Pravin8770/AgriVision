import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import streamlit as st
import json
import requests
from PIL import Image
import numpy as np
import pandas as pd
try:
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    st.warning("TensorFlow not available - Disease Detection feature disabled")
from deep_translator import GoogleTranslator
import google.generativeai as genai
from docx import Document  
from dotenv import load_dotenv

from chatbot.llama_chatbot import LlamaAgriChatbot 


st.set_page_config(page_title="Plant Health App", layout="wide")

# Cache model to optimize performance
@st.cache_resource
def load_plant_model():
    if TF_AVAILABLE:
        model_path = os.path.join(os.path.dirname(__file__), "plant_disease_prediction_model.h5")
        return load_model(model_path, compile=False)
    return None

# Load static JSON data once
@st.cache_resource
def load_json(file_name):
    with open(os.path.join(os.path.dirname(__file__), file_name), "r") as file:
        return json.load(file)

working_dir = os.path.dirname(os.path.abspath(__file__))

# Initialize Model & Data
model = load_plant_model()
class_indices = load_json("class_indices.json")
recommendations = load_json("recommendations.json")
market_data = load_json("market.json")
fertilizer_stores = load_json("maharashtra_fertilizer_stores.json")
crop_npk_data = load_json("crop_npk.json")


# API Configuration
OPENWEATHER_API_KEY = "e904ee2b79326aba2a44970e6ddce3d1"
WEATHER_URL = "http://api.openweathermap.org/data/2.5/weather"


# Language Selection
translator = GoogleTranslator(source='en', target='mr')
LANG_DICT = {
    "English": {
        "Home": "Home",
        "Disease Detection": "Disease Detection",
        "Market Analysis": "Market Analysis",
        "Weather Analysis": "Weather Analysis",
        "Nearby Stores": "Nearby Stores",
        "Crop Prediction": "Crop Prediction",
        "AI Assistant": "AI Assistant"
    },
    "मराठी": {
        "Home": "मुख्यपृष्ठ",
        "Disease Detection": "रोग शोध",
        "Market Analysis": "बाजार विश्लेषण",
        "Weather Analysis": "हवामान विश्लेषण",
        "Nearby Stores": "नजीकची खते दुकाने",
        "Crop Prediction": "पिक अंदाज",
        "AI Assistant": "कृषी सल्लागार" 
    }
}



language = st.sidebar.radio("Select Language / भाषा निवडा", ["English", "मराठी"])
# def t(text): return LANG_DICT[language].get(text, translator.translate(text, dest="mr").text) if language == "मराठी" else text

def t(text):
    if language == "मराठी":
        # First check if it's in our predefined dictionary
        if text in LANG_DICT["English"]:
            index = list(LANG_DICT["English"].values()).index(text)
            key = list(LANG_DICT["English"].keys())[index]
            return LANG_DICT["मराठी"][key]
        
        # If not in dictionary, use translator
        try:
            translated_text = translator.translate(text)
            # Apply custom replacements
            translated_text = translated_text.replace("अॅप", "अ‍ॅप").replace("प्लांट", "वनस्पती").replace("हेल्थ", "आरोग्य")
            return translated_text
        except:
            return text  # Fallback to original text if translation fails
    return text


# Utility Functions
def get_weather(city):
    params = {"q": city, "appid": OPENWEATHER_API_KEY, "units": "metric"}
    try:
        response = requests.get(WEATHER_URL, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return None

def load_and_preprocess_image(image, target_size=(224, 224)):
    img = image.resize(target_size)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_image_class(model, image):
    preprocessed_img = load_and_preprocess_image(image)
    predictions = model.predict(preprocessed_img)
    return class_indices.get(str(np.argmax(predictions, axis=1)[0]), "Unknown")

# UI Components
def navbar():
    st.markdown(f"<nav style='background:#008000;padding:10px;text-align:center;color:white;'>"
                f"<h1>{t('Plant Health App')}</h1></nav>", unsafe_allow_html=True)

def footer():
    st.markdown(f"<footer style='background:#222;padding:10px;text-align:center;color:white;'>"
                f"&copy; 2025 {t('Plant Health App. All Rights Reserved.')}</footer>", unsafe_allow_html=True)

# Pages
def home():
    navbar()
    st.header(t("Welcome to Plant Health App 🌿"))
    st.write(t("This app helps in plant disease detection, market analysis for crops, and weather updates."))
    footer()

def disease_detection():
    navbar()
    st.title(t('🌿 Plant Disease Detection'))
    
    if not TF_AVAILABLE:
        st.error("This feature requires TensorFlow which is not available in your environment.")
        st.info("Please install TensorFlow or use Python 3.10/3.11 for full functionality.")
        footer()
        return
    
    uploaded_image = st.file_uploader(t("📤 Upload an image..."), type=["jpg", "jpeg", "png"])
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image.resize((150, 150)), caption=t("Uploaded Image"))
        if st.button(t('🔍 Classify')):
            with st.spinner(t("Processing...")):
                prediction = predict_image_class(model, image)
                st.success(f'✅ {t("Prediction")}: {t(prediction)}')
                if prediction in recommendations:
                    for key, value in recommendations[prediction].items():
                        st.write(f"**{t(key)}**: {t(value)}")
    footer()

# def market_analysis():
#     navbar()
#     st.title(t("📊 Market Analysis"))
#     selected_crop = st.selectbox(t("Select a Crop:"), list(market_data["crops"].keys()))
#     if selected_crop:
#         crop_prices = market_data["crops"][selected_crop]
#         min_price, max_price = crop_prices["min_price"], crop_prices["max_price"]
#         st.success(f'{t("Current market price range for")} {t(selected_crop)}: ₹{min_price} - ₹{max_price}')
#     footer()

# load_dotenv()
# load_dotenv(dotenv_path="plant disease zip/key.env")
# genai.configure(api_key=os.getenv("AIzaSyAr5YMqdSKP411z4jGDYNoa76eDbwoDFfA"))
# api_key = os.getenv("GOOGLE_API_KEY")

# if not api_key:
#     raise ValueError("API Key not found. Please check your .env file.")

# API_KEY = "AIzaSyAr5YMqdSKP411z4jGDYNoa76eDbwoDFfA"
# headers = {
#     "Authorization": f"Bearer {API_KEY}",
#     "Content-Type": "application/json"
# }

# # Sample request
# response = requests.get("https://generativelanguage.googleapis.com/v1/models/gemini-pro", headers=headers)

# if response.status_code != 200:
#     st.error("Invalid API Key or request!")
    
# genai.configure(api_key=api_key)
# # Function to extract Q&A pairs
# def extract_qa_pairs(doc_path):
#     doc = Document(doc_path)
#     qa_pairs = {}
#     question = None

#     for para in doc.paragraphs:
#         text = para.text.strip()
#         if text.startswith("Q:"):
#             question = text[2:].strip()
#         elif text.startswith("A:") and question:
#             answer = text[2:].trip()
#             qa_pairs[question] = answer
#             question = None

#     return qa_pairs

# # Load Q&A pairs
# qa_pairs = extract_qa_pairs("ANSWERS.docx")

# # Function to generate AI response
# def get_gemini_response(prompt):
#     model = genai.GenerativeModel("gemini-pro")
#     response = model.generate_content(prompt)
#     return response.text

# # Streamlit UI
# st.title("💬 Gemini Q&A Chatbot")
# st.write("Ask questions based on your dataset")

# user_input = st.text_input("Ask a question:")

# if user_input:
#     if user_input in qa_pairs:
#         st.write(f"✅ **Answer:** {qa_pairs[user_input]}")
#     else:
#         st.write("🤖 **AI Response:**")
#         response = get_gemini_response(user_input)
#         st.write(response)






def market_analysis():
    navbar()
    st.title(t("📊 Market Analysis"))

    selected_crop = st.selectbox(t("Select a Crop:"), list(market_data["crops"].keys()))

    if selected_crop:
        crop_prices = market_data["crops"][selected_crop]
        min_price, max_price = crop_prices["min_price"], crop_prices["max_price"]

        st.success(f'{t("Current market price range for")} {t(selected_crop)}: ₹{min_price} - ₹{max_price}')

        # Plot diagonal line from min to max price
        fig, ax = plt.subplots(figsize=(8, 3))  # Adjusted figure size
        ax.plot([0, 1], [min_price, max_price], marker='o', color='black', linestyle='-', linewidth=3)

        # Set Min/Max labels on X-axis
        ax.set_xticks([0, 1])
        ax.set_xticklabels([t("Min"), t("Max")])  
        ax.set_ylabel(t("Price (₹)"))
        ax.set_title(t(f"Market Price Trend for {selected_crop}"))
        ax.grid(axis='y', linestyle='--', alpha=0.6)

        st.pyplot(fig, use_container_width=False)  # Prevent full-screen width

    footer()


# def weather_analysis():
#     navbar()
#     st.title(t("🌦 Weather Analysis"))
#     city = st.text_input(t("Enter city name:"))
#     if city and st.button(t("Get Weather")):
#         with st.spinner(t("Fetching data...")):
#             weather_data = get_weather(city)
#         if weather_data:
#             st.success(f"{t('Weather in')} {city}:")
#             st.write(f"🌡 {t('Temperature')}: {weather_data['main']['temp']}°C")
#             st.write(f"💧 {t('Humidity')}: {weather_data['main']['humidity']}%")
#             st.write(f"💨 {t('Wind Speed')}: {weather_data['wind']['speed']} m/s")
#             st.write(f"☁ {t('Weather')}: {t(weather_data['weather'][0]['description'].capitalize())}")
#         else:
#             st.error(t("City not found or API error."))
#     footer()

def weather_analysis():
    navbar()
    st.title(t("🌦 Weather Analysis"))
    
    city = st.text_input(t("Enter city name:"))
    
    if city and st.button(t("Get Weather")):
        with st.spinner(t("Fetching data...")):
            weather_data = get_weather(city)
        
        if weather_data:
            temp = weather_data['main']['temp']
            humidity = weather_data['main']['humidity']
            wind_speed = weather_data['wind']['speed']
            description = t(weather_data['weather'][0]['description'].capitalize())

            # Define Colors Based on Temperature
            temp_color = "#FF5733" if temp > 30 else "#3498DB"
            
            st.success(f"{t('Weather in')} {city}")

            # Layout using Columns
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(f"### 🌡 **{t('Temperature')}**")
                st.markdown(f"<h2 style='color:{temp_color};'>{temp}°C</h2>", unsafe_allow_html=True)

            with col2:
                st.markdown(f"### 💧 **{t('Humidity')}**")
                st.markdown(f"<h2 style='color:#1ABC9C;'>{humidity}%</h2>", unsafe_allow_html=True)

            with col3:
                st.markdown(f"### 💨 **{t('Wind Speed')}**")
                st.markdown(f"<h2 style='color:#F39C12;'>{wind_speed} m/s</h2>", unsafe_allow_html=True)

            # Weather Condition Box
            st.markdown(
                f"<div style='background:#2C3E60;padding:10px;border-radius:10px;color:white;text-align:center;'>"
                f"<h3>☁ {t('Weather Condition')}</h3>"
                f"<h4>{description}</h4></div> <br>",
                unsafe_allow_html=True,
            )

        else:
            st.error(t("City not found or API error."))

    footer()




def nearby_stores():
    navbar()
    st.title(t("🛒 Nearby Fertilizer Stores"))
    stores_path = os.path.join(working_dir, "maharashtra_fertilizer_stores.json")
    with open(stores_path, "r") as file:
        store_data = json.load(file)
    selected_city = st.selectbox(t("🌍 Select City:"), list(store_data.keys()))
    if st.button(t("🔍 Search Stores")):
        st.subheader(f"{t('Stores in')} {selected_city}:")
        for store in store_data[selected_city]:
            st.write(f"{store['name']} - {store['address']}")
        st.map(pd.DataFrame(store_data[selected_city]))
    footer()

def crop_prediction():
    navbar()
    st.title(t("🌾 Crop Prediction"))

    N = st.number_input(t("Enter Nitrogen (N) value"), min_value=0, max_value=300)
    P = st.number_input(t("Enter Phosphorus (P) value"), min_value=0, max_value=300)
    K = st.number_input(t("Enter Potassium (K) value"), min_value=0, max_value=300)

    if st.button(t("🔍 Predict Best Crops")):
        matching_crops = [
            crop for crop, values in crop_npk_data.items()
            if values["N"][0] <= N <= values["N"][1] and
               values["P"][0] <= P <= values["P"][1] and
               values["K"][0] <= K <= values["K"][1]
        ]

        if matching_crops:
            st.success(f"{t('Best Crops for given NPK values')}: 🌾 {', '.join(map(t, matching_crops))}")
        else:
            st.warning(t("No exact match found. Consider adjusting NPK values."))

    footer()

# Load environment variables (API keys)
load_dotenv()

@st.cache_resource
def get_llama_chatbot():
    """Initialize and cache the LLaMA chatbot instance"""
    api_key = os.environ.get("NVIDIA_API_KEY")
    return LlamaAgriChatbot(api_key=api_key)

def ai_assistant():
    navbar()
    st.title(t("🤖 AI Agricultural Assistant"))
    
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Get chatbot instance (cached)
    try:
        chatbot = get_llama_chatbot()
    except Exception as e:
        st.error(f"Failed to initialize chatbot: {str(e)}")
        st.warning(t("Make sure the NVIDIA_API_KEY is set in your environment or .env file"))
        footer()
        return
    
    # Chat input
    if prompt := st.chat_input(t("Ask about farming, crops, or agricultural practices...")):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
            
        # Display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Stream the response
            formatted_prompt = chatbot.agricultural_prompt(prompt)
            
            try:
                # Use spinner while waiting for initial response
                with st.spinner(t("Thinking...")):
                    for response_chunk in chatbot.get_response(formatted_prompt):
                        full_response += response_chunk
                        message_placeholder.write(full_response + "▌")
                
                # Final response without cursor
                message_placeholder.write(full_response)
                
                # Add response to chat history
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # Add a button to clear chat history
    if st.button(t("Clear Conversation")):
        st.session_state.messages = []
        chatbot.clear_history()
        st.rerun()
        
    footer()

# Sidebar Navigation
page = st.sidebar.radio(t("Navigate"), list(LANG_DICT[language].values()))
if page == t("Home"):
    home()
elif page == t("Disease Detection"):
    disease_detection()
elif page == t("Market Analysis"):
    market_analysis()
elif page == t("Weather Analysis"):
    weather_analysis()
elif page == t("Nearby Stores"):
    nearby_stores()
elif page == t("Crop Prediction"):
    crop_prediction()
elif page == t("AI Assistant"):
    ai_assistant()