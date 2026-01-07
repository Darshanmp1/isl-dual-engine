# ISL Text-to-Sign Translation (Simplified)

## 🎯 Overview
**Ultra-simplified ISL translator** using direct video mapping. All code merged into just **2 files**!

## 📁 File Structure

```
model_training/src/
├── isl_mapper.py     # All core logic (400+ lines, everything you need)
├── app.py            # Streamlit UI (80 lines)
└── README.md         # This file
```

**That's it!** No more scattered files.

## 🔧 What's in isl_mapper.py?

One file contains everything:
- ✅ **Dataset Mapper** - Load videos from directory
- ✅ **Video Generator** - Create sign language videos
- ✅ **Preprocessing** - Generate labels.csv
- ✅ **Evaluation** - Coverage stats
- ✅ **Visualizations** - Charts and plots
- ✅ **Utilities** - Helper functions

## 🚀 Quick Start

### 1. Add Videos
```bash
# Put sign videos in data/raw_videos/
# Format: Hello.mp4, A.mp4, Thank_You.mp4
```

### 2. Run App
```bash
streamlit run app.py
```

### 3. (Optional) CLI Commands
```bash
# Setup dataset
python isl_mapper.py

# Or use as module
python -c "from isl_mapper import setup_dataset; setup_dataset()"
```

## 💻 Usage Examples

### As a Module
```python
from isl_mapper import build_video_map, text_to_sign_video

# Load videos
video_map = build_video_map()

# Generate sign video
clip = text_to_sign_video("hello world", video_map)
clip.write_videofile("output.mp4")
```

### Preprocessing
```python
from isl_mapper import generate_labels_csv, validate_videos

# Generate labels from videos
generate_labels_csv()

# Validate videos
valid, invalid = validate_videos()
print(f"Valid: {valid}, Invalid: {len(invalid)}")
```

### Evaluation
```python
from isl_mapper import evaluate_dataset_coverage, print_evaluation_report

# Evaluate coverage
results = evaluate_dataset_coverage()
print_evaluation_report(results)
```

### Visualizations
```python
from isl_mapper import generate_all_visualizations

# Create all charts
generate_all_visualizations()
```

## 🎨 How It Works

```
User Input → Clean Text → For each word:
                          ├─ Word in dataset? → Play word.mp4
                          └─ Not found? → Spell: a.mp4 + b.mp4 + c.mp4
```

## 📊 Dataset Format

**Video Files** (in `data/raw_videos/`):
- Letters: `A.mp4`, `B.mp4`, ..., `Z.mp4`
- Numbers: `0.mp4`, `1.mp4`, ..., `9.mp4`
- Words: `Hello.mp4`, `Thank_You.mp4`, etc.

**Auto-generated** (`data/labels.csv`):
```csv
video_name,label,label_lowercase
Hello.mp4,Hello,hello
Thank_You.mp4,Thank You,thank you
```

## 📦 Dependencies

```
streamlit
moviepy
pandas
matplotlib
numpy
```

No PyTorch, transformers, or heavy ML libraries!

## 🎯 Benefits

- 🪶 **Tiny codebase**: Just 2 files
- ⚡ **Fast**: No model loading/inference
- 🎯 **Simple**: Easy to understand and modify
- 📚 **Complete**: All features in one place
- 🔧 **Maintainable**: One file to rule them all

## 🚨 Key Points

- ✅ No training required
- ✅ No BERT/transformer models
- ✅ No frame extraction
- ✅ Direct video concatenation
- ✅ 100% deterministic output

## 📞 Integration

Both files can be imported and used anywhere:

```python
# In your main app
from model_training.src.isl_mapper import text_to_sign_video, build_video_map

video_map = build_video_map()
clip = text_to_sign_video("hello", video_map)
```

---

**Last Updated**: January 2026  
**Total Files**: 2  
**Lines of Code**: ~500  
**Approach**: Direct Mapping (No ML)
