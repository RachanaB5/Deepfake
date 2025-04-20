# app.py
import streamlit as st
import requests
from PIL import Image
import os

# Initialize Streamlit
st.set_page_config(page_title="Deepfake Detector", layout="wide")
st.title("🔍 Deepfake Detection")

def call_detection_api(image_file):
    try:
        # First upload the file and get metadata
        response = requests.post(
            "http://127.0.0.1:3200/detect",
            files={"file": image_file},
            timeout=15
        )
        
        if response.status_code != 200:
            error_detail = response.json().get("error", response.text)
            st.error(f"Backend Error ({response.status_code}): {error_detail}")
            return None
            
        result = response.json()
        
        # Then download the heatmap image
        heatmap_response = requests.get(
            f"http://127.0.0.1:3200/heatmap/{result['heatmap_filename']}",
            timeout=15
        )
        
        if heatmap_response.status_code == 200:
            # Save heatmap to temp file
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp.write(heatmap_response.content)
                result["heatmap_path"] = tmp.name
            return result
        else:
            st.error(f"Failed to get heatmap: {heatmap_response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        st.error("Backend server unavailable. Please ensure the Flask server is running.")
    except requests.exceptions.Timeout:
        st.error("Request timed out. Try a smaller image or check your connection.")
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
    return None

# File uploader
uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(uploaded_file, caption="Original Image", use_container_width=True)
    
    if st.button("Analyze"):
        with st.spinner("Analyzing..."):
            result = call_detection_api(uploaded_file)
            
            if result:
                try:
                    heatmap = Image.open(result["heatmap_path"])
                    with col2:
                        st.image(heatmap, caption="Analysis Results", use_container_width=True)
                        status = "❌ Fake" if result["is_fake"] else "✅ Authentic"
                        st.success(f"{status} (Confidence: {result['confidence']:.2f})")
                    os.unlink(result["heatmap_path"])  # Clean up
                except Exception as e:
                    st.error(f"Failed to display results: {str(e)}")

st.caption("Note: Requires Flask backend running on port 3200 (run 'python flask_api.py')")