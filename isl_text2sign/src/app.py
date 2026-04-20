# model_training/src/app.py
# Streamlit ISL Translator - Text to Sign Language Video

import streamlit as st

from isl_text2sign.src.isl_mapper import build_video_map, text_to_sign_video

st.set_page_config(page_title="ISL Translator", page_icon="🧏‍♀️", layout="centered")

# Header
st.title("🧏‍♀️ Indian Sign Language Translator")
st.markdown("### Text → Sign Language Video")
st.write("Type any English sentence and see its sign language translation.")

# Load video map with caching
@st.cache_resource
def load_video_map():
    return build_video_map()

video_map = load_video_map()

# Input section
user_input = st.text_input("Enter text to translate:", placeholder="e.g., Hello my name is John")
translate_btn = st.button("🎥 Translate to Sign", use_container_width=True)

# Translation
if translate_btn:
    if not user_input.strip():
        st.warning("⚠️ Please enter some text to translate!")
    else:
        with st.spinner("🔄 Generating sign language video..."):
            clip = text_to_sign_video(user_input, video_map)

            if clip:
                output_path = "output.mp4"
                clip.write_videofile(
                    output_path, 
                    codec="libx264", 
                    fps=24,
                    preset="ultrafast",
                    bitrate="1500k",
                    audio=False,
                    threads=8,
                    ffmpeg_params=["-crf", "23"],
                    logger=None
                )
                
                st.success("✅ Translation complete!")
                st.video(output_path)
            else:
                st.error("❌ No matching signs found in dataset.")
