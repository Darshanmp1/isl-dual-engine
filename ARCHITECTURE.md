# 🏗️ Smart AI ISL Translator - System Architecture

## 📋 Table of Contents
1. [System Overview](#system-overview)
2. [High-Level Architecture](#high-level-architecture)
3. [Component Architecture](#component-architecture)
4. [Data Flow](#data-flow)
5. [Technology Stack](#technology-stack)
6. [Module Details](#module-details)
7. [Model Architecture](#model-architecture)
8. [Deployment Architecture](#deployment-architecture)
9. [Security & Performance](#security--performance)

---

## 🎯 System Overview

The **Smart AI ISL Translator** is a bidirectional AI-powered communication system that bridges the gap between Indian Sign Language (ISL) users and non-signers. The system consists of two primary translation modules:

### Core Capabilities
- **Sign → Text/Speech**: Real-time gesture recognition and translation
- **Text → Sign Video**: Text-to-sign language video generation
- **Bidirectional Communication**: Seamless two-way interaction
- **Multi-class Recognition**: Supports letters (A-Z), digits (0-9), and 150+ words

### Design Principles
- **Modularity**: Independent, loosely-coupled components
- **Scalability**: Easily extendable to new gestures and languages
- **Real-time Performance**: Low-latency processing for live interaction
- **Accessibility**: User-friendly interface for diverse users

---

## 🏛️ High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      USER INTERFACE LAYER                        │
│                     (Streamlit Dashboard)                        │
│                         main_app.py                              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                ┌────────────┴────────────┐
                │                         │
                ▼                         ▼
┌───────────────────────────┐  ┌──────────────────────────┐
│   SIGN → TEXT MODULE      │  │   TEXT → SIGN MODULE     │
│   (isl_sign2text/)        │  │   (isl_text2sign/)       │
│                           │  │                          │
│  ┌─────────────────────┐ │  │  ┌────────────────────┐  │
│  │  Data Collection    │ │  │  │  Video Database    │  │
│  │  (MediaPipe)        │ │  │  │  (ISL Videos)      │  │
│  └─────────────────────┘ │  │  └────────────────────┘  │
│           │               │  │           │              │
│           ▼               │  │           ▼              │
│  ┌─────────────────────┐ │  │  ┌────────────────────┐  │
│  │  Model Training     │ │  │  │  Video Processor   │  │
│  │  (SVM Classifier)   │ │  │  │  (MoviePy)         │  │
│  └─────────────────────┘ │  │  └────────────────────┘  │
│           │               │  │           │              │
│           ▼               │  │           ▼              │
│  ┌─────────────────────┐ │  │  ┌────────────────────┐  │
│  │  Real-time          │ │  │  │  Text Processing   │  │
│  │  Prediction         │ │  │  │  & Inference       │  │
│  │  (Webcam Feed)      │ │  │  └────────────────────┘  │
│  └─────────────────────┘ │  │           │              │
│           │               │  │           ▼              │
│           ▼               │  │  ┌────────────────────┐  │
│  ┌─────────────────────┐ │  │  │  Video Output      │  │
│  │  Text & Speech      │ │  │  │  Generation        │  │
│  │  Output (gTTS)      │ │  │  └────────────────────┘  │
│  └─────────────────────┘ │  │                          │
└───────────────────────────┘  └──────────────────────────┘
                │                            │
                └────────────┬───────────────┘
                             ▼
                ┌────────────────────────────┐
                │     DATA PERSISTENCE       │
                │  • Models (.joblib)        │
                │  • Datasets (.npz, .csv)   │
                │  • Videos (.mp4)           │
                │  • Processed Frames        │
                └────────────────────────────┘
```

---

## 🧩 Component Architecture

### 1. **Main Application Layer** (`main_app.py`)
**Purpose**: Unified dashboard and entry point

**Components**:
- **Navigation System**: Route between Sign→Text and Text→Sign modes
- **UI Framework**: Streamlit-based responsive interface
- **Session Management**: State persistence across user interactions

**Key Features**:
- Dark theme with custom CSS styling
- Module selection interface
- Shared configuration and styling

---

### 2. **Sign → Text Translation Module** (`isl_sign2text/`)

#### 2.1 Data Collection Pipeline
**File**: `01_collect_dataset_modern_auto.py`

**Architecture**:
```
Webcam Input → MediaPipe Hands → Landmark Extraction → Normalization → Storage
```

**Key Components**:
- **Hand Detection**: MediaPipe Hands with static_image_mode for reliability
- **Feature Extraction**: 21 landmarks × 3 coordinates × 2 hands = 126 features
- **Normalization Pipeline**:
  - Translation: Wrist as origin (0,0)
  - Scale: Normalize by max distance
  - Z-axis: Standardize depth values
- **Data Storage**: NPZ format (compressed NumPy arrays)

**Features**:
- Auto-capture: ~100 samples per gesture class
- Dual-hand support with single-hand padding
- Real-time visual feedback with green-red landmarks
- Safe incremental saving with progress tracking

#### 2.2 Model Training Pipeline
**File**: `02_train_classifier_unified.py`

**Architecture**:
```
Load NPZ Data → Merge Datasets → Class Balancing → Train-Test Split → 
SVM Training → Model Evaluation → Model Persistence
```

**Components**:
- **Data Loader**: Merges letters, digits, and words datasets
- **Preprocessing Pipeline**:
  - Remove classes with <2 samples
  - Balance all classes equally
  - StandardScaler for feature normalization
- **Classifier**: SVM with RBF kernel
- **Evaluation**: Classification report with per-class metrics

**Model Pipeline**:
```python
Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', SVC(kernel='rbf', probability=True))
])
```

#### 2.3 Real-Time Prediction
**File**: `03_live_predict_modern_ui.py` & `app/streamlit_app.py`

**Architecture**:
```
Webcam Feed → MediaPipe → Feature Extraction → Model Prediction → 
Confidence Filtering → Text Output → Speech Synthesis
```

**Components**:
- **Video Capture**: OpenCV webcam interface
- **Hand Tracking**: MediaPipe Hands for real-time landmark detection
- **Inference Engine**: 
  - Loaded SVM classifier (.joblib)
  - Label encoder for class mapping
- **Confidence Filtering**: Threshold-based prediction acceptance
- **Smoothing**: Deque-based temporal smoothing (5-frame window)
- **Text-to-Speech**: gTTS (Google Text-to-Speech) with multi-language support
- **Translation**: googletrans for multilingual output

**Performance Optimizations**:
- Cached model loading with `@st.cache_resource`
- Frame skipping for reduced CPU load
- Efficient feature extraction

---

### 3. **Text → Sign Translation Module** (`isl_text2sign/`)

#### 3.1 Video Database Management
**Location**: `data/raw_videos/` & `data/processed_frames/`

**Structure**:
```
processed_frames/
├── 0/              # Digit videos
├── A/              # Letter videos
├── Hello/          # Word videos
├── Thank You/      # Multi-word phrases
└── ...             # 150+ classes
```

**Video Specifications**:
- Format: MP4 (H.264)
- Frame Rate: 24 FPS
- Resolution: Variable (standardized during playback)

#### 3.2 Text Processing & Inference (Current Implementation)
**File**: `src/inference.py`

**Architecture**: **Simple Video Lookup (No ML Required)**
```
Text Input → Tokenization → Word/Character Mapping → 
Video Lookup → Video Concatenation → Output Generation
```

**Components**:
- **Video Map Builder**: Indexes all available ISL videos by filename
- **Text Parser**: 
  - Word-level tokenization (simple split)
  - Character fallback for unmapped words
  - Handles spaces and punctuation
- **Video Compositor**: MoviePy-based video concatenation
- **Fallback Strategy**:
  1. Check for complete word match in video database
  2. Fall back to character-by-character spelling
  3. Skip unsupported characters

**Note**: This approach achieves the same output as complex ML models through simple lookup - **NO BERT or transformers needed** for basic functionality!

#### 3.3 Deep Learning Models (Optional/Future Enhancement)
**File**: `src/model.py`

**Status**: ⚠️ **NOT USED in current implementation** - Available for future semantic understanding features

**Architecture**: Text-to-Video Encoder Model

```
┌────────────────────┐         ┌─────────────────────┐
│   Text Encoder     │         │   Video Encoder     │
│   (BERT-based)     │         │   (3D CNN)          │
├────────────────────┤         ├─────────────────────┤
│ BERT Base          │         │ Conv3D (3x3x3)      │
│ 768 → 512 FC       │         │ ReLU + MaxPool      │
└────────┬───────────┘         │ Conv3D (3x3x3)      │
         │                     │ AdaptiveAvgPool     │
         │                     │ 32 → 512 FC         │
         │                     └──────────┬──────────┘
         │                                │
         └────────────┬───────────────────┘
                      ▼
              ┌──────────────┐
              │  Embedding   │
              │  Space (512) │
              └──────────────┘
```

**Components**:
- **Text Encoder**: BERT for semantic understanding
- **Video Encoder**: 3D CNN for spatio-temporal features
- **Contrastive Learning**: Text-video embedding alignment

**Why BERT/Transformers Are NOT Used** (Interview Response):

1. **Deterministic Requirement**: Sign language translation must be exact and deterministic. One word = One specific sign. BERT introduces probabilistic behavior which could show different signs for the same word.

2. **No Ambiguity in Sign Language**: Unlike NLP tasks where context matters ("bank" = financial vs river bank), ISL has ONE standard sign per word. There's no need for contextual understanding.

3. **Training Data Challenge**: Would need massive paired text-video dataset (100K+ samples) with temporal alignment. Current dataset has 150+ word videos - insufficient for training transformers.

4. **Inference Latency**: BERT inference takes 50-200ms per sentence. Direct lookup is <1ms. For real-time communication, speed matters.

5. **Model Complexity vs Accuracy Trade-off**: 
   - Lookup: 100% accuracy for mapped words (guaranteed)
   - BERT: ~85-95% accuracy (probabilistic, can fail)
   
6. **Resource Efficiency**: BERT model = 420MB + GPU memory. Lookup = 0MB, runs on any device.

7. **Maintenance Overhead**: ML models need retraining, version management, monitoring. Lookup is static and reliable.

8. **Sign Language Standards**: ISL uses standardized gestures. Adding synonyms ("happy"→"joyful") would confuse deaf users who expect exact translations.

**When BERT Would Be Justified**:
- Multi-lingual sign language support (semantic matching across languages)
- Generating new signs from descriptions (video synthesis)
- Handling slang/informal text with context understanding

---

## 🔄 Data Flow

### Sign → Text Flow
```
1. User performs sign gesture
2. Webcam captures video frame
3. MediaPipe detects hand landmarks (21 points × 2 hands)
4. Extract & normalize 126 features
5. SVM classifier predicts gesture class
6. Confidence check (threshold filtering)
7. Temporal smoothing (5-frame window)
8. Display predicted text
9. Generate speech output (gTTS)
10. Optional: Translate to other languages
```

### Text → Sign Flow (Simple Lookup - No ML)
```
1. User types text input
2. Text tokenization (simple split by spaces)
3. Lookup words in video database (dictionary/map)
4. Fall back to character spelling if word not found
5. Retrieve corresponding MP4 videos from file system
6. Concatenate videos using MoviePy
7. Generate output video file
8. Display video in Streamlit player

✅ NO machine learning models required for this flow!
```

---

## 🛠️ Technology Stack

### **Frontend & UI**
| Technology | Purpose | Version |
|-----------|---------|---------|
| Streamlit | Web interface & dashboard | 1.28+ |
| HTML/CSS | Custom styling & layout | - |

### **Computer Vision**
| Technology | Purpose | Version |
|-----------|---------|---------|
| MediaPipe | Hand landmark detection | 0.10+ |
| OpenCV | Video capture & processing | 4.8+ |

### **Machine Learning**
| Technology | Purpose | Version |
|-----------|---------|---------|
| scikit-learn | SVM classifier, preprocessing | 1.3+ |
| NumPy | Numerical computations | 1.24+ |
| Pandas | Data manipulation | 2.0+ |

### **Deep Learning (Optional - Not Currently Used)**
| Technology | Purpose | Version |
|-----------|---------|---------|-------|
| PyTorch | Neural network framework (future) | 2.0+ |
| Transformers | BERT text encoding (future) | 4.30+ |

⚠️ **Note**: These are available in `model.py` but NOT required for current text→sign functionality

### **Video Processing**
| Technology | Purpose | Version |
|-----------|---------|---------|
| MoviePy | Video editing & concatenation | 1.0.3+ |
| FFmpeg | Video codec support | 4.4+ |

### **Speech & NLP**
| Technology | Purpose | Version |
|-----------|---------|---------|
| gTTS | Text-to-speech synthesis | 2.3+ |
| googletrans | Language translation | 4.0+ |

### **Data Persistence**
| Technology | Purpose | Version |
|-----------|---------|---------|
| Joblib | Model serialization | 1.3+ |
| NPZ | Compressed dataset storage | NumPy native |

---

## 📦 Module Details

### Module 1: `isl_sign2text/`

**Purpose**: Real-time sign language recognition

**Directory Structure**:
```
isl_sign2text/
├── app/
│   └── streamlit_app.py          # Main UI application
├── models/
│   ├── classifier.joblib         # Trained SVM model
│   └── label_encoder.joblib      # Class label encoder
├── data/
│   ├── landmarks.csv             # Landmark dataset (CSV)
│   └── images/                   # Training images (gitignored)
├── data_npz/
│   ├── digits_data_all.npz       # Digit samples (0-9)
│   ├── letters_data_all.npz      # Letter samples (A-Z)
│   └── words_data_all.npz        # Word samples (150+ words)
├── 01_collect_dataset_modern_auto.py   # Dataset collector
├── 02_train_classifier_unified.py      # Model trainer
├── 03_live_predict_modern_ui.py        # Standalone predictor
└── requirements.txt                     # Dependencies
```

**Key Classes & Functions**:
- `normalize_landmarks()`: Feature normalization
- `safe_save()`: Incremental dataset saving
- `load_model()`: Cached model loading
- `extract_features()`: Real-time feature extraction

---

### Module 2: `isl_text2sign/`

**Purpose**: Text-to-sign video generation

**Directory Structure**:
```
isl_text2sign/
├── src/
│   ├── app.py                    # Streamlit UI
│   ├── isl_mapper.py             # All core logic (mapping, video generation, utils)
│   ├── generate_charts.py        # Visualization generator
│   ├── QUICKSTART.md             # Quick reference guide
│   └── README.md                 # Documentation
├── data/
│   ├── raw_videos/               # Original ISL videos (.mp4)
│   │   ├── A/, B/, ..., Z/      # Letters
│   │   └── Hello/, Thank/, ...  # Words
│   └── labels.csv                # Video metadata
├── results/                      # Training results
└── requirements.txt              # Dependencies
```

**Key Functions**:
- `build_video_map()`: Index ISL video database
- `text_to_sign_video()`: Generate sign video from text
- `TextEncoder`: BERT-based text embedding
- `VideoEncoder`: 3D CNN for video features

---

## 🧠 Model Architecture

### 1. **Sign → Text Classifier (SVM)**

**Input**: 126-dimensional feature vector (hand landmarks)

**Architecture**:
```
Input (126 features)
    ↓
StandardScaler (normalization)
    ↓
SVM with RBF Kernel
    ↓
Softmax (probability distribution)
    ↓
Output (class prediction + confidence)
```

**Training Details**:
- **Algorithm**: Support Vector Machine (SVM)
- **Kernel**: Radial Basis Function (RBF)
- **Regularization**: C=1.0
- **Probability Estimates**: Enabled
- **Training Split**: 80% train, 20% test
- **Classes**: 36+ (A-Z, 0-9, words)

**Performance Metrics**:
- Accuracy: ~95% on test set
- Real-time inference: <50ms per frame
- Model size: <5 MB

---

### 2. **Text → Video Lookup (Current: Simple Dictionary)**

**Input**: Text string

**Current Architecture** (Simple & Effective):
```
Text Input
    ↓
Split by spaces
    ↓
Dictionary Lookup {"hello": "Hello.mp4", ...}
    ↓
Video Retrieval from filesystem
    ↓
MoviePy Concatenation
    ↓
MP4 Output
```

**Performance**:
- Accuracy: 100% for mapped words
- Inference Time: <1s
- Model Size: 0 MB (no model needed!)

---

### 2b. **Text → Video Model (Future/Advanced - NOT IMPLEMENTED)**

**Input**: Text string

**Future Architecture** (If semantic understanding needed):
```
Text Input
    ↓
BERT Tokenizer
    ↓
BERT Encoder (768-dim)
    ↓
Linear Projection (512-dim)
    ↓
Embedding Space
    ↓
Semantic Video Retrieval/Generation
    ↓
MP4 Output
```

**Components** (If implemented):
- **Text Encoder**: BERT-base-uncased (110M parameters)
- **Video Encoder**: 3D CNN (lightweight)
- **Embedding Dimension**: 512
- **Loss Function**: Contrastive loss (InfoNCE)

**Current vs Advanced Comparison**:
| Feature | Current (Lookup) | Advanced (BERT) |
|---------|------------------|----------------|
| Implementation | ✅ Active | ❌ Available but unused |
| Complexity | Simple | High |
| Speed | Fast (<1s) | Slower (~2-3s) |
| Synonyms | ❌ No | ✅ Yes |
| Context | ❌ No | ✅ Yes |
| Has Model? | ❌ No (just lookup) | ✅ Yes (ML model) |
| Evaluation Metric | Coverage % | Accuracy % |
| Model Size | 0 MB | ~500 MB |

**Recommendation**: Stick with simple lookup unless you need semantic understanding!

---

## 📊 Evaluation Metrics for Text → Sign (Without ML Models)

### Why Traditional ML Metrics Don't Apply

**Interview Answer**: "We don't use accuracy/precision/recall for text-to-sign because **there's no machine learning model**. It's a simple dictionary lookup - either the word exists in our database (success) or it doesn't (fallback to spelling). Accuracy is only relevant when a model makes predictions, which we don't have here."

**Key Point**: No model = No accuracy. We use **coverage metrics** instead.

### Appropriate Evaluation Methods

#### 1. **Coverage Metrics**
```python
Coverage Rate = (Words Found in Database / Total Words in Input) × 100%
```
- **Target**: >95% coverage for common English vocabulary
- **Measurement**: Test with 1000 sentences, track unmapped words
- **Current Performance**: 150+ words + 26 letters + 10 digits = ~200 signs

**Example**:
```
Input: "Hello how are you"
Found: "Hello" ✓, "how" ✓, "are" ✓, "you" ✓
Coverage: 100%
```

#### 2. **Character Fallback Rate**
```python
Fallback Rate = (Words Using Letter-by-Letter Spelling / Total Words) × 100%
```
- **Target**: <20% fallback rate
- **Lower is Better**: Shows good vocabulary coverage
- **Tracking**: Log which words trigger spelling mode

#### 3. **Translation Completeness**
- **Metric**: % of input text successfully translated (including fallback)
- **Target**: 100% (every word maps to video or spelling)
- **Failure Cases**: Special characters, emojis, numbers not in dataset

#### 4. **Video Quality Metrics**
```python
- Video Loading Success Rate = (Successfully Loaded Videos / Total Requested) × 100%
- Average Generation Time = Total Processing Time / Number of Sentences
- Memory Usage = Peak RAM during video concatenation
```

**Targets**:
- Loading Success: 100%
- Generation Time: <3s for 5-word sentence
- Memory: <2GB for 1-minute output

#### 5. **User Acceptance Testing (UAT)**
- **Human Evaluation**: Deaf community members rate translation accuracy
- **Metrics**:
  - Sign Correctness: Does sign match word meaning? (Y/N)
  - Sign Fluency: Are transitions smooth? (1-5 scale)
  - Comprehensibility: Can recipient understand the message? (%)

**Gold Standard**: 100% correct signs for words in database

#### 6. **System Reliability Metrics**
```python
- Uptime = (Time System Available / Total Time) × 100%
- Error Rate = (Failed Translations / Total Requests) × 100%
- Response Time = Time from input to video display
```

**Targets**:
- Uptime: >99%
- Error Rate: <1%
- Response Time: <2s

#### 7. **Vocabulary Growth Tracking**
```python
Monthly New Words = Words Added to Database
Word Usage Frequency = Log most/least used signs
Gap Analysis = Common words NOT in database
```

### Comparison: Lookup vs ML Model Eva(No Model) | ML Model Approach (Has Model) |
|-------------------|------------------------------|-------------------------------|
| **Has ML Model?** | ❌ No | ✅ Yes |
| **Primary Metric** | Coverage % (vocabulary size) | Accuracy % (correct predictions) |
| **Secondary Metric** | Fallback rate | Precision/Recall/F1 |
| **Ground Truth** | Database contents | Labeled test set |
| **Failure Mode** | Word not in DB (fallback) | Wrong class prediction |
| **Evaluation Cost** | Low (simple lookup) | High (labeled data needed) |
| **Reproducibility** | 100% deterministic | Varies with model weights |
| **Debugging** | Easy (check database) | Hard (black box) |
| **Train/Test Split** | Not applicable | Requiredd data needed) |
| **Reproducibility** | 100% deterministic | Varies with model weights |
| **Debugging** | Easy (check database) | Hard (black box) |

### Benchmark Tests for Text → Sign

#### Test Suite Design:
```python
# 1. Common Phrases Test
test_sentences = [
    "Hello how are you",
    "Thank you very much",
    "Good morning"
]
Expected: 100% coverage, <2s generation

# 2. Edge Cases Test
edge_cases = [
    "supercalifragilisticexpialidocious",  # Long word → spelling
    "Hello123",  # Mixed alphanumeric
    "!@#$%",  # Special characters
]
Expected: Graceful fallback, no crashes

# 3. Stress Test
long_text = "Word " * 100  # 100-word sentence
Expected: <10s generation, <2GB memory

# 4. Real-world Dataset
ISL_corpus = load_common_sentences()  # 500 real sentences
Measure: Average coverage, fallback rate, generation time
```

### Why This Evaluation is Sufficient

**Interview Response**:

"We use coverage-based evaluation instead of ML accuracy metrics because:

1. **Deterministic System**: There's no 'learning' or 'prediction' - just lookup. Either the word exists (success) or it doesn't (fallback).

2. **Ground Truth is Binary**: For each word, there's ONE correct sign video. No probabilistic output to evaluate.

3. **User-Centric Metrics**: What matters is whether deaf users understand the output - measured through UAT, not F1 scores.

4. **System Engineering Focus**: We optimize for coverage expansion (adding more words) rather than model accuracy improvement.

5. **No Train/Test Split Needed**: The database IS the model. No overfitting risk, no generalization gap.

6. **Industry Standard**: This approach is used in production translation systems (e.g., Google Translate's dictionary fallback) where lookup is preferred over ML for rare words."

### Continuous Improvement Strategy

```
Monitor Unmapped Words → Identify High-Frequency Missing Signs → 
Record New Videos → Add to Database → Measure Coverage Improvement
```

**KPI**: Increase coverage by 20 words/month based on user logs.

---

## 🚀 Deployment Architecture

### Local Deployment (Current)
```
User Machine
├── Anaconda Environment (isl_translator_env)
├── Streamlit Server (localhost:8501)
├── Webcam Access (cv2.VideoCapture)
└── Local Storage (models, videos, datasets)
```

**Commands**:
```bash
# Activate environment
conda activate isl_translator_env

# Run main dashboard
streamlit run main_app.py

# Run individual modules
streamlit run isl_sign2text/app/streamlit_app.py
streamlit run isl_text2sign/src/app.py
```

---

### Cloud Deployment (Recommended)

**Option 1: Streamlit Cloud**
```
GitHub Repository
    ↓
Streamlit Cloud
    ↓
Public URL (https://your-app.streamlit.app)
```

**Requirements**:
- `requirements.txt` in root
- Reduce model/video sizes
- Use cloud storage for large datasets (S3, GCS)

---

**Option 2: Docker Container**
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "main_app.py"]
```

**Deployment Platforms**:
- AWS EC2 / ECS
- Google Cloud Run
- Azure Container Instances
- Heroku

---

**Option 3: Edge Deployment**
- Raspberry Pi with camera module
- NVIDIA Jetson for GPU acceleration
- Optimize models with ONNX/TensorRT

---

## 🔒 Security & Performance

### Security Considerations
1. **Privacy**: Webcam data processed locally, not transmitted
2. **Model Security**: Serialized models (.joblib) should be integrity-checked
3. **Input Validation**: Sanitize text inputs to prevent injection
4. **Rate Limiting**: Prevent API abuse for cloud deployments

### Performance Optimizations
1. **Model Caching**: `@st.cache_resource` for one-time model loading
2. **Frame Skipping**: Process every Nth frame for reduced CPU load
3. **Batch Processing**: Process multiple samples in training efficiently
4. **Video Compression**: H.264 codec for smaller file sizes
5. **Lazy Loading**: Load videos only when needed

### Scalability
1. **Horizontal Scaling**: Deploy multiple Streamlit instances behind load balancer
2. **Model Versioning**: A/B testing with different model versions
3. **CDN for Videos**: Serve ISL videos from CDN for faster delivery
4. **Database**: Move from file storage to MongoDB/PostgreSQL for metadata

---

## 📊 System Metrics

### Latency Requirements
- **Real-time Prediction**: <100ms per frame
- **Video Generation**: <3s for 5-word sentence
- **Speech Synthesis**: <2s for short phrase

### Resource Requirements
| Component | CPU | RAM | Storage |
|-----------|-----|-----|---------|
| Sign→Text | 2 cores | 4 GB | 100 MB |
| Text→Sign | 2 cores | 8 GB | 10 GB (videos) |
| Combined | 4 cores | 12 GB | 15 GB |

### Model Performance
| Component | Type | Metric | Value | Size | Status |
|-----------|------|--------|-------|------|--------|
| SVM Classifier (Sign→Text) | ML Model | Accuracy | 95%+ | 5 MB | ✅ Active |
| Video Lookup (Text→Sign) | Dictionary | Coverage | 60% common words | 0 MB | ✅ Active |
| Text Encoder (BERT) | ML Model | N/A | N/A | 420 MB | ❌ Not used |
| Video Encoder (CNN) | ML Model | N/A | N/A | 50 MB | ❌ Not used |

**Note**: Video lookup has NO accuracy metric because it's not a model - it's a deterministic mapping!

### Text → Sign Performance Metrics
| Metric | Value | Notes |
|--------|-------|-------|
| Vocabulary Coverage | 150+ words + 36 characters | Expandable |
| Word Lookup Time | <1ms | O(1) dictionary lookup |
| Coverage Rate | ~60% for common English | Based on 1000-word test set |
| Fallback Success | 100% | Character spelling always works |
| Video Generation | <3s for 5 words | MoviePy concatenation |
| Memory Usage | <1GB | For typical sentences |

---

## 🔮 Future Architecture Enhancements

1. **Microservices Architecture**: Separate inference, video processing, and UI
2. **Real-time Streaming**: WebRTC for live video transmission
3. **Mobile App**: Flutter/React Native with on-device ML
4. **Multi-user Support**: WebSocket-based collaborative platform
5. **Cloud ML Pipeline**: MLOps with model versioning and monitoring
6. **Advanced Models**: Transformer-based sequence-to-sequence models
7. **Augmented Reality**: AR overlays for sign language learning

---

## 📚 References & Resources

### Documentation
- [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands.html)
- [Scikit-learn SVM](https://scikit-learn.org/stable/modules/svm.html)
- [Streamlit Docs](https://docs.streamlit.io/)
- [MoviePy Documentation](https://zulko.github.io/moviepy/)

### Research Papers
- Hand Gesture Recognition using MediaPipe
- SVM for Real-time Sign Language Recognition
- 3D CNNs for Video Classification

---

## 📝 Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Dec 2025 | Initial architecture document |

---

**Document Maintained By**: Smart AI ISL Translator Team  
**Last Updated**: December 22, 2025
