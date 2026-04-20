# Smart AI-Powered ISL Translator
> "Bridging the communication gap with a modern, real-time Flask-React architecture."

> **Live Demo**: [Application URL](https://your-app-link-here.com)

[![System UI](https://img.shields.io/badge/UI-Modern-brightgreen)](https://github.com/Darshanmp1/isl-translator)
[![Backend](https://img.shields.io/badge/Backend-Flask-blue)](https://github.com/Darshanmp1/isl-translator/tree/main/backend)
[![Frontend](https://img.shields.io/badge/Frontend-React%20%2B%20Vite-61dafb)](https://github.com/Darshanmp1/isl-translator/tree/main/frontend)

---

## Project Overview
The Smart AI ISL Translator is a high-performance, bidirectional communication system designed to facilitate interaction for the hearing-impaired community. The platform utilizes a decoupled architecture to ensure scalability and real-time responsiveness.

- **Automated User Interface**: A responsive frontend built with React 18 and Vite, featuring low-latency webcam streaming and dynamic model selection.
- **Hybrid Machine Learning Backend**: A Flask-based API serving a dual-model inference engine (SVM and 1D-CNN) for robust gesture recognition.
- **Bidirectional Translation Workflow**:
    - **Sign-to-Text**: Real-time hand landmark extraction using MediaPipe, processed via 126-dimensional feature vectors.
    - **Text-to-Sign**: Sequential animation of Indian Sign Language (ISL) video assets with a character-level fallback mechanism.
- **Speech Synthesis Integration**: High-quality multilingual audio output powered by gTTS.

---

## System Architecture
The application architecture is designed for modularity and high throughput between the vision processing layer and the user interface.

```mermaid
graph TD
    subgraph "Frontend (React + Vite)"
        UI[User Interface]
        WC[Webcam Streamer]
        MT[Model Toggle]
        UI --> WC
        UI --> MT
    end

    subgraph "Backend API (Flask)"
        API[Flask API]
        MP[MediaPipe Processor]
        subgraph "Inference Engine"
            CLF_SVM[SVM Classifier]
            CLF_CNN[1D-CNN Classifier]
        end
        API --> MP
        MP --> API
        API -- "predict?model=..." --> CLF_SVM
        API -- "predict?model=..." --> CLF_CNN
    end

    subgraph "Storage and Assets"
        MODELS[Models: .joblib / .keras]
        VIDS[ISL Videos .mp4]
        API <--> VIDS
        API <--> MODELS
    end

    WC -- "Base64 Frames" --> API
    API -- "Predictions + Speech" --> UI
```

---

## Model Architectures
The system implements a tiered classification strategy to balance inference speed and analytical depth.

| Engine | Methodology | Feature Set | Performance Characteristics |
| :--- | :--- | :--- | :--- |       
| **SVM** | Support Vector Machine | RBF Kernel, 126-dim landmarks | Optimized for low-latency CPU inference. |
| **CNN** | Deep Learning | 1D-Convolutional Neural Network | Enhanced pattern recognition for complex hand shapes. |

> [!IMPORTANT]
> **Fault Tolerance**: The backend includes a graceful fallback mechanism. If deep learning dependencies are unavailable, the system automatically redirects all traffic to the SVM engine to maintain continuous service.

---

## Installation and Deployment

### 1. Repository Setup
Clone the localized repository and navigate into the project directory:
```bash
git clone https://github.com/Darshanmp1/isl-dual-engine.git
cd isl-dual-engine
```

### 2. Environment Configuration
It is recommended to use an isolated environment for dependency management:
```bash
conda create -n isl_env python=3.10
conda activate isl_env
pip install -r backend/requirements.txt
```

### 3. Dataset Configuration
Due to the significant size of the video and landmark datasets, they are hosted externally. Follow these steps to initialize the asset library:

#### Text-to-Sign Assets (ISL Videos)
- **Dataset**: Indian Sign Language Animated Videos
- **Source**: [Kaggle - Indian Sign Language Animated Videos](https://www.kaggle.com/datasets/koushikchouhan/indian-sign-language-animated-videos)
- **Deployment**: Extract the downloaded MP4 files to `isl_text2sign/data/raw_videos/`.

#### Sign-to-Text Training Data (Optional)
- **Purpose**: Required only if you intend to retrain the SVM or CNN models.
- **Inference**: For standard use, pre-trained binaries are already provided in `isl_sign2text/models/`.

### 4. Backend Execution
Initialize the Flask server:
```bash
cd backend
python app.py
```

### 5. Frontend Execution
Launch the React development server:
```bash
cd frontend
npm install
npm run dev
```

> Access the application via the local proxy at `http://localhost:5173`.

### 6. Deployment via Docker (Recommended for Production)
The application is fully containerized using Docker and Docker Compose. This ensures environment consistency across different systems.

#### Prerequisites
- Docker Desktop or Docker Engine installed.
- Ensure the ISL datasets are downloaded to the correct local path (see section 3).

#### Launching the Containers
From the project root, execute the following command:
```bash
docker-compose up --build
```

#### Access Points
- **Frontend**: `http://localhost:80`
- **Backend API**: `http://localhost:5000`

> **Note**: The Docker setup automatically mounts your local sign videos into the container for real-time sequencing.

---

## Directory Structure
```text
isl-translator/
├── backend/                # Flask API and inference logic
├── frontend/               # React/Vite user interface
├── isl_sign2text/          # Machine learning core and training pipelines
│   ├── models/             # Serialized model binaries (.joblib / .keras)
│   └── 03_train_cnn.py     # CNN training script
├── isl_text2sign/          # Video data and assets
└── ARCHITECTURE.md         # Technical design documentation
```
 
---

## Model Training Pipeline (Advanced)
For researchers and developers looking to extend the dataset or retrain the engines, the following utilities are provided in the `isl_sign2text/` directory:

### 1. Data Collection
- **`01_collect_dataset_modern_auto.py`**: Facilitates the recording of new sign language gestures via webcam. It automatically extracts 126-dimensional landmarks and saves them as normalized `.npz` files for training.

### 2. Model Training
- **`02_train_classifier_unified.py`**: The primary script for training the Support Vector Machine (SVM) classifier. It includes data cleaning and performance evaluation metrics.
- **`03_train_cnn.py`**: Specifically used for training the 1D-Convolutional Neural Network (CNN) engine using the consolidated landmark datasets.

### 3. Independent Verification
- **`03_live_predict_modern_ui.py`**: A standalone real-time inference tool. Use this to verify model accuracy in a controlled environment before deploying to the Flask/React web stack.

---

## Technology Stack
- **Frontend**: React 18, Vite, CSS3
- **Backend**: Flask, Flask-CORS, gTTS
- **Machine Learning**: MediaPipe, scikit-learn, TensorFlow/Keras, NumPy
---

## 📜 License
Licensed under the MIT License.
