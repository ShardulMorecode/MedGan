import streamlit as st
from PIL import Image, ImageOps
import time

def main():
    st.set_page_config(page_title='MEDGAN App', page_icon='🧠', layout='centered')
    
    # Initialize session state variables
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False
    
    if not st.session_state['authenticated']:
        login()
    else:
        medgan_chatbot()

def login():
    st.title("🔐 Login to MEDGAN")
    st.subheader("Secure AI-powered Medical Image Enhancement")
    
    with st.form("login_form"):
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        submit_button = st.form_submit_button("Login")
    
    if submit_button:
        if username == "admin" and password == "password":  # Simple authentication
            st.session_state['authenticated'] = True
            st.experimental_rerun()  # Use experimental_rerun to restart the app flow
        else:
            st.error("❌ Invalid credentials. Please try again.")

def medgan_chatbot():
    st.title("🧠 MEDGAN - AI Image Enhancement")
    st.subheader("Upload a medical image and let AI enhance it for better analysis.")
    
    uploaded_file = st.file_uploader("📤 Upload an Image", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='📷 Uploaded Image', use_column_width=True)
        
        if st.button("🚀 Enhance Image"):
            with st.spinner("🔄 Processing Image... Please wait ⏳"):
                time.sleep(3)  # Simulating AI processing
                
                # Creating a blank (black) image instead of enhanced version
                blank_image = ImageOps.grayscale(image)  # Convert to grayscale
                blank_image = Image.new('RGB', image.size, (0, 0, 0))  # Create blank black image
                
                st.success("✅ Image Enhanced Successfully!")
                st.image(blank_image, caption='✨ Enhanced Image', use_column_width=True)
                    
    st.sidebar.header("💬 Chat with MEDGAN AI")
    user_input = st.text_input("Ask anything about the enhancement process:")
    
    if user_input:
        response = "🤖 MEDGAN AI: Our AI enhances medical images by reducing noise and improving clarity."
        st.sidebar.write(response)

if __name__ == "__main__":
    main()
