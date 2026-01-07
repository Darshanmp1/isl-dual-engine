🤖 Smart AI-Powered Indian Sign Language (ISL) Translator
“Bridging the communication gap between speech and silence”


🧩 Abstract

The Smart AI ISL Translator is an intelligent real-time system designed to bridge the communication gap between hearing-impaired individuals and the general population.
It performs two-way translation between Indian Sign Language (ISL) gestures and text/audio output, as well as converting text input into sign language animations.

This dual functionality is implemented using Streamlit, MediaPipe, and Machine Learning models (SVM/CNN) trained on a custom dataset of ISL gestures.
The system helps in education, accessibility, and daily communication for hearing-impaired communities.

📘 Introduction

Sign language is a visual means of communication used primarily by the hearing and speech-impaired community. However, communication barriers still exist between sign language users and non-signers.

The Smart AI ISL Translator aims to:

Recognize real-time hand gestures using webcam input and translate them into text/speech.

Convert typed text or sentences into corresponding ISL sign videos, enabling two-way interaction.

This system uses AI, Deep Learning, and Computer Vision techniques with tools like:

MediaPipe Hands for gesture tracking

Scikit-learn / TensorFlow models for classification

Streamlit for interactive user interface

gTTS (Google Text-to-Speech) for voice output

⚙️ Implementation Overview
🔹 Modules:

Sign → Text Translator

Captures hand gestures through webcam

Detects and tracks landmarks using MediaPipe

Extracts features, normalizes them, and classifies using a trained SVM model

Converts predicted gesture to text and speech output

Text → Sign Translator

Accepts typed English text

Checks for available ISL word/letter videos in the dataset

Plays matching animations in sequence using MoviePy

Provides a smooth video-based ISL output

Unified Dashboard (main_app.py)

Acts as the entry point for both translators

Lets the user choose between the two translation modes

🧠 Technologies Used
Category	Tools / Frameworks
Programming Language	Python 3.9+
Frontend/UI	Streamlit
Machine Learning	scikit-learn, TensorFlow/Keras
Computer Vision	MediaPipe, OpenCV
Speech	gTTS (Google Text-to-Speech)
Video Processing	MoviePy
Data Handling	NumPy, Pandas
Storage Format	.joblib (Model), .mp4 (ISL videos)
📁 Project Structure
```
isl-translator/
│
├── main_app.py                                    # Unified Dashboard (Entry Point)
├── README.md
├── .gitignore
│
├── isl_sign2text/                                 # Sign → Text Translator Module
│   ├── app/
│   │   └── streamlit_app.py                       # Streamlit app for sign recognition
│   ├── models/
│   │   ├── classifier.joblib                      # Trained SVM classifier
│   │   └── label_encoder.joblib                   # Label encoder for classes
│   ├── data/
│   │   ├── landmarks.csv                          # Collected hand landmarks
│   │   └── images/                                # (gitignored) Training images
│   ├── data_npz/
│   │   ├── digits_data_all.npz                    # (gitignored) Digit dataset
│   │   ├── letters_data_all.npz                   # (gitignored) Letter dataset
│   │   └── words_data_all.npz                     # (gitignored) Word dataset
│   ├── 01_collect_dataset_modern_auto.py          # Dataset collection script
│   ├── 02_train_classifier_unified.py             # Model training script
│   ├── 03_live_predict_modern_ui.py               # Real-time prediction script
│   └── requirements.txt
│
├── model_training/                                # Text → Sign Translator Module
│   ├── src/
│   │   ├── app.py                                 # Streamlit app for text-to-sign
│   │   ├── model.py                               # Deep learning model architecture
│   │   ├── dataset_loader.py                      # Data loading utilities
│   │   ├── train.py                               # Model training pipeline
│   │   ├── inference.py                           # Inference & video generation
│   │   ├── evaluate.py                            # Model evaluation metrics
│   │   ├── preprocess.py                          # Data preprocessing
│   │   └── utils.py                               # Helper functions
│   ├── models/
│   │   ├── text_to_video_model.pt                 # (gitignored) Trained model
│   │   └── checkpoints/                           # (gitignored) Training checkpoints
│   ├── data/
│   │   ├── labels.csv                             # Dataset labels
│   │   ├── raw_videos/                            # (gitignored) ISL videos - Download from Kaggle
│   │   │   └── .gitkeep
│   │   └── processed_frames/                      # (gitignored) Extracted frames
│   │       └── .gitkeep
│   ├── results/
│   └── requirements.txt
│
└── screenshots/                                    # Application screenshots
    ├── Screenshot 2025-11-23 115543.png
    ├── Screenshot 2025-11-23 115642.png
    ├── Screenshot 2025-11-23 115828.png
    └── Screenshot 2025-11-24 164327.png
```

🧰 Setup Instructions
1️⃣ Clone the Repository
```bash
git clone https://github.com/Darshanmp1/isl-translator.git
cd isl-translator
```

2️⃣ Download Required Datasets

**⚠️ Important:** The datasets are not included in the repository due to their large size. Download them separately:

**For Text → Sign Translation (ISL Videos):**
- **Dataset:** Indian Sign Language Animated Videos
- **Source:** [Kaggle - Indian Sign Language Animated Videos](https://www.kaggle.com/datasets/koushikchouhan/indian-sign-language-animated-videos)
- **Download location:** Extract to `model_training/data/raw_videos/`

**For Sign → Text Training (Optional - only if retraining):**
- The model is pre-trained and included in `isl_sign2text/models/`
- If you want to retrain, collect your own dataset using `01_collect_dataset_modern_auto.py`

3️⃣ Create Virtual Environment (Recommended)
```bash
python -m venv venv
venv\Scripts\activate      # for Windows
source venv/bin/activate   # for macOS/Linux
```

4️⃣ Install Dependencies

**For Sign → Text module:**
```bash
cd isl_sign2text
pip install -r requirements.txt
```

**For Text → Sign module:**
```bash
cd ../model_training
pip install -r requirements.txt
```

**(Or install all at once:)**
```bash
pip install streamlit mediapipe opencv-python moviepy joblib scikit-learn gtts googletrans==4.0.0-rc1
```

🧪 Execution Steps
🔹 Option 1: Launch Unified Dashboard
streamlit run main_app.py


This will open the home page where you can choose between Sign→Text or Text→Sign modules.

🔹 Option 2: Run Sign → Text Translator Directly
streamlit run isl_sign2text/app/streamlit_app.py


This module:

Opens webcam

Detects hand gestures

Displays live predictions

Converts text output to voice

🔹 Option 3: Run Text → Sign Translator Directly
streamlit run model_training/src/app.py


This module:

Accepts English text input

Plays matching ISL word/letter videos sequentially

Automatically handles missing words by spelling them letter-by-letter

📸 Expected Output
Module	Output
Sign → Text	Live webcam feed → Predicted ISL letter/word → Text + Audio output
Text → Sign	Input sentence → Video display of corresponding ISL gestures
Dashboard	One-click access to both modules with simple UI
🧠 Key Features

🖐 Real-time gesture recognition using MediaPipe

🧠 Trained SVM/CNN model for accurate classification

🗣 Voice output in multiple languages using Google TTS

🎞 Video animation playback for ISL gestures

🔄 Two-way translation (Sign↔Text↔Sign)

🪶 Lightweight Streamlit UI for fast prototyping and deployment

📊 Model Performance

**Sign → Text Recognition (SVM Classifier)**
- **Training Accuracy:** 95-98%
- **Validation Accuracy:** 92-95%
- **Classes Supported:** A-Z (26 letters), 0-9 (10 digits), 100+ common words
- **Real-time FPS:** 25-30 frames per second
- **Model Size:** <5MB (lightweight deployment)

**Text → Sign Generation (Deep Learning)**
- **Dataset:** 200+ ISL video animations
- **Training Loss:** Converged using Cosine Embedding Loss
- **Similarity Score:** High text-video feature alignment (see histogram below)
- **Video Quality:** 720p MP4 format
- **Supported Words:** 150+ common ISL words + letter-by-letter spelling
- **Response Time:** <2 seconds per sentence

![Text-Video Similarity Distribution](model_training/results/similarity_histogram.png)
*Distribution of cosine similarity scores between text and video embeddings*

> **Note:** Accuracy may vary based on lighting conditions, camera quality, and hand positioning. Best results achieved with clear background and proper lighting.

🧾 Execution Flow Summary
Step	Process
1	Capture gesture using webcam
2	Extract hand landmarks using MediaPipe
3	Normalize and preprocess coordinates
4	Classify gesture using trained SVM/CNN model
5	Display prediction + Convert to speech
6	(Optional) Convert text back to ISL video
📸 Application Screenshots

### Unified Dashboard
![Main Dashboard](screenshots/Screenshot%202025-11-24%20164327.png)

### Sign to Text Translation
![Sign to Text](screenshots/Screenshot%202025-11-23%20115543.png)

### Text to Sign Translation
![Text to Sign](screenshots/Screenshot%202025-11-23%20115642.png)

### Real-time Gesture Recognition
![Gesture Recognition](screenshots/Screenshot%202025-11-23%20115828.png)

🧑‍💻 Future Enhancements

Add gesture-to-gesture contextual translation

Expand dataset with more word-level ISL signs

Deploy on cloud (Render/Heroku/AWS) with real-time multilingual support

Mobile application development for Android/iOS

🏆 Conclusion

This project successfully demonstrates an AI-driven Indian Sign Language Translation System capable of bridging the communication barrier between hearing-impaired and non-signing individuals.
Through real-time hand gesture recognition, text-to-sign visualization, and speech synthesis, it provides an accessible and practical communication tool.

💡 Execution Commands Summary
Module	Command
Unified Dashboard	streamlit run main_app.py
Sign → Text	streamlit run isl_sign2text/app/streamlit_app.py
Text → Sign	streamlit run model_training/src/app.py

