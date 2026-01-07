# Quick Reference Guide

## Files Structure (Simplified to 2 files!)

```
model_training/src/
├── isl_mapper.py       ← All core logic (dataset, video generation, utils)
├── app.py              ← Streamlit UI
├── generate_charts.py  ← Generate visualizations
└── README.md           ← Documentation
```

## Quick Commands

### Run Streamlit App
```bash
cd model_training/src
streamlit run app.py
```

### Generate Visualizations
```bash
python generate_charts.py
```

Charts generated:
- ✅ Word Coverage Analysis
- ✅ Video Generation Performance
- ✅ Dataset Composition

Saved to: `model_training/results/`

### Setup Dataset (Optional)
```bash
python isl_mapper.py
```

### Use as Python Module
```python
from isl_mapper import build_video_map, text_to_sign_video

# Load videos
video_map = build_video_map()

# Generate sign video
clip = text_to_sign_video("hello world", video_map)
clip.write_videofile("output.mp4")
```

## Generate Custom Visualizations

```python
from isl_mapper import generate_all_visualizations

# Generate all charts
results = generate_all_visualizations()

# Or generate individually
from isl_mapper import generate_coverage_plot, generate_performance_plot

# Custom coverage test
test_results = [
    {'sentence': 'test1', 'coverage': 75},
    {'sentence': 'test2', 'coverage': 50}
]
generate_coverage_plot(test_results, 'custom_coverage.png')

# Custom performance data
word_counts = [1, 4, 7, 8]
gen_times = [0.29, 2.67, 8.77, 5.28]
generate_performance_plot(word_counts, gen_times, 'custom_performance.png')
```

## What Each File Does

### isl_mapper.py (13KB, ~400 lines)
Contains EVERYTHING:
- ISLDatasetMapper class
- build_video_map()
- text_to_sign_video()
- generate_labels_csv()
- evaluate_dataset_coverage()
- generate_all_visualizations()
- generate_coverage_plot()
- generate_performance_plot()
- All utility functions

### app.py (4KB, ~90 lines)
Streamlit UI:
- Import from isl_mapper
- Display UI
- Call text_to_sign_video()
- Show results

### generate_charts.py (1KB)
Quick script to generate all visualizations

## That's It!

No more scattered files. Everything you need is in these files.
