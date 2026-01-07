# isl_mapper.py
# All-in-one ISL Text-to-Sign mapping module
# Combines: dataset_loader, model, inference, utils, preprocess, evaluate, visualizations

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from moviepy import VideoFileClip, concatenate_videoclips, ColorClip

# ===============================
# 🔧 CONFIGURATION
# ===============================
BASE_DIR = Path(__file__).resolve().parents[1]
VIDEO_DIR = BASE_DIR / "data" / "raw_videos"
LABELS_CSV = BASE_DIR / "data" / "labels.csv"
RESULTS_DIR = BASE_DIR / "results"

# ===============================
# 🧰 UTILITY FUNCTIONS
# ===============================
def clean_text(text):
    """Remove special characters and lowercase text."""
    return re.sub(r"[^a-zA-Z0-9\s]", "", text).lower().strip()

def validate_video_path(video_path):
    """Check if video file exists and is valid."""
    if not os.path.exists(video_path):
        return False
    if not video_path.lower().endswith('.mp4'):
        return False
    return True

def get_dataset_stats(video_map):
    """Get statistics about the dataset."""
    letters = [k for k in video_map.keys() if len(k) == 1]
    words = [k for k in video_map.keys() if len(k) > 1]
    
    return {
        'total_signs': len(video_map),
        'letters': len(letters),
        'words': len(words),
        'letter_list': sorted(letters),
        'word_list': sorted(words)
    }

def format_match_info(input_text, video_map):
    """Format information about which words matched and which need spelling."""
    words = clean_text(input_text).split()
    matched = [w for w in words if w in video_map]
    spelled = [w for w in words if w not in video_map]
    
    info = []
    if matched:
        info.append(f"✅ Matched words: {', '.join(matched)}")
    if spelled:
        info.append(f"🔤 Spelled out: {', '.join(spelled)}")
    
    return "\n".join(info) if info else "No valid words found"

# ===============================
# 📦 DATASET MAPPER
# ===============================
class ISLDatasetMapper:
    """Simple dataset mapper that creates word/letter to video path mappings."""
    
    def __init__(self, video_dir=None):
        self.video_dir = Path(video_dir) if video_dir else VIDEO_DIR
        self.video_map = self._build_video_map()
    
    def _build_video_map(self):
        """Build mapping of words/letters to video file paths."""
        if not self.video_dir.exists():
            os.makedirs(self.video_dir, exist_ok=True)
            print(f"⚠️ Created empty video folder: {self.video_dir}")
            return {}
        
        video_map = {}
        for video_file in os.listdir(self.video_dir):
            if video_file.lower().endswith('.mp4'):
                # Convert filename to lowercase key (remove .mp4 and convert spaces/underscores)
                key = os.path.splitext(video_file)[0].lower().replace('_', ' ').strip()
                video_map[key] = str(self.video_dir / video_file)
        
        return video_map
    
    def get_video_path(self, word):
        """Get video path for a word/letter."""
        return self.video_map.get(word.lower())
    
    def has_word(self, word):
        """Check if word exists in dataset."""
        return word.lower() in self.video_map
    
    def has_letter(self, letter):
        """Check if letter exists in dataset."""
        return letter.lower() in self.video_map
    
    def get_all_words(self):
        """Get list of all available words."""
        return list(self.video_map.keys())
    
    def reload(self):
        """Reload video mappings from directory."""
        self.video_map = self._build_video_map()
        return len(self.video_map)

# ===============================
# 🎬 VIDEO GENERATION
# ===============================
def build_video_map(video_dir=None):
    """Build and return video mapping dictionary."""
    mapper = ISLDatasetMapper(video_dir)
    return mapper.video_map

def add_pause(duration=0.3, size=(640, 480), color=(255, 255, 255)):
    """Return a blank clip used as a short pause between signs."""
    return ColorClip(size=size, color=color, duration=duration)

def text_to_sign_video(text, video_map):
    """
    Convert text into concatenated ISL video using word fallback.
    - Step 1: Check if entire phrase exists in dataset (e.g., "thank you")
    - Step 2: Check individual words (e.g., "thank", "you")
    - Step 3: Fallback to letter-by-letter spelling
    - All case-insensitive
    """
    # Normalize input to lowercase for matching
    text_lower = text.lower().strip()
    clips = []
    
    # First, try to match the entire phrase (e.g., "thank you")
    if text_lower in video_map:
        clips.append(VideoFileClip(video_map[text_lower]))
        return concatenate_videoclips(clips, method="compose") if clips else None
    
    # If not, split into words and process each
    words = clean_text(text).split()
    
    for i, word in enumerate(words):
        word_lower = word.lower().strip()
        
        # Try perfect word match first (case-insensitive)
        if word_lower in video_map:
            clips.append(VideoFileClip(video_map[word_lower]))
        else:
            # Fallback: spell word letter-by-letter
            for char in word_lower:
                if char.isalnum() and char in video_map:
                    clips.append(VideoFileClip(video_map[char]))

        # Pause between words only (not after the last word)
        if i < len(words) - 1:
            clips.append(add_pause(0.3))

    if not clips:
        return None

    try:
        return concatenate_videoclips(clips, method="compose")
    except Exception as e:
        print(f"❌ Error generating video: {e}")
        return None

# ===============================
# 📝 PREPROCESSING
# ===============================
def generate_labels_csv(video_dir=None, output_csv=None):
    """Generate labels.csv from video filenames."""
    video_dir = Path(video_dir) if video_dir else VIDEO_DIR
    output_csv = output_csv or LABELS_CSV
    
    if not video_dir.exists():
        print(f"⚠️ Video directory not found: {video_dir}")
        os.makedirs(video_dir, exist_ok=True)
        return
    
    data = []
    for file in os.listdir(video_dir):
        if file.lower().endswith('.mp4'):
            label = os.path.splitext(file)[0].replace('_', ' ')
            data.append({
                'video_name': file,
                'label': label,
                'label_lowercase': label.lower()
            })
    
    if not data:
        print(f"⚠️ No .mp4 videos found in {video_dir}")
        return
    
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"✅ labels.csv saved to {output_csv} ({len(data)} videos)")
    return df

def validate_videos(video_dir=None):
    """Validate that all videos are accessible."""
    video_dir = Path(video_dir) if video_dir else VIDEO_DIR
    
    if not video_dir.exists():
        print(f"⚠️ Directory not found: {video_dir}")
        return 0, []
    
    valid_count = 0
    invalid_files = []
    
    for file in os.listdir(video_dir):
        if file.lower().endswith('.mp4'):
            file_path = video_dir / file
            if os.path.getsize(file_path) > 0:
                valid_count += 1
            else:
                invalid_files.append(file)
    
    return valid_count, invalid_files

# ===============================
# 📊 EVALUATION
# ===============================
def evaluate_dataset_coverage(video_dir=None, test_sentences=None):
    """Evaluate dataset coverage on test sentences."""
    if test_sentences is None:
        test_sentences = [
            "hello",
            "my name is john",
            "thank you",
            "how are you",
            "good morning"
        ]
    
    mapper = ISLDatasetMapper(video_dir)
    stats = get_dataset_stats(mapper.video_map)
    
    results = {
        'stats': stats,
        'tests': []
    }
    
    for sentence in test_sentences:
        words = clean_text(sentence).split()
        matched = sum(1 for w in words if mapper.has_word(w))
        coverage = (matched / len(words) * 100) if words else 0
        
        results['tests'].append({
            'sentence': sentence,
            'total_words': len(words),
            'matched_words': matched,
            'coverage': coverage
        })
    
    return results

def print_evaluation_report(results):
    """Print formatted evaluation report."""
    print("=" * 60)
    print("📊 Dataset Coverage Evaluation")
    print("=" * 60)
    
    stats = results['stats']
    print(f"\n✅ Total signs available: {stats['total_signs']}")
    print(f"   • Letters: {stats['letters']}")
    print(f"   • Words: {stats['words']}")
    
    print(f"\n📋 Available letters: {', '.join(stats['letter_list'])}")
    
    if stats['word_list']:
        print(f"\n📋 Sample words (first 20):")
        for word in stats['word_list'][:20]:
            print(f"   • {word}")
        if len(stats['word_list']) > 20:
            print(f"   ... and {len(stats['word_list']) - 20} more")
    
    print(f"\n🧪 Test Coverage Results:")
    print("-" * 60)
    for test in results['tests']:
        print(f"\n'{test['sentence']}'")
        print(f"  Words: {test['total_words']} | Matched: {test['matched_words']} | Coverage: {test['coverage']:.0f}%")
    
    avg_coverage = np.mean([t['coverage'] for t in results['tests']])
    print(f"\n📈 Average Coverage: {avg_coverage:.1f}%")
    print("=" * 60)

# ===============================
# 📊 VISUALIZATIONS
# ===============================
def generate_dataset_stats_plot(stats, save_path):
    """Generate pie chart showing dataset composition."""
    plt.figure(figsize=(8, 6))
    
    labels = ['Letters', 'Words']
    sizes = [stats['letters'], stats['words']]
    colors = ['#4CAF50', '#2196F3']
    explode = (0.1, 0)
    
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90)
    plt.title(f'ISL Dataset Composition\n(Total: {stats["total_signs"]} signs)')
    plt.axis('equal')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

def generate_coverage_plot(test_results, save_path):
    """Generate bar chart showing word coverage for test sentences."""
    sentences = [r['sentence'] for r in test_results]
    coverage = [r['coverage'] for r in test_results]
    
    avg_coverage = np.mean(coverage)
    colors = ['#4CAF50' if c >= avg_coverage else '#FFA726' for c in coverage]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(sentences)), coverage, color=colors)
    
    for i, (bar, cov) in enumerate(zip(bars, coverage)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{int(cov)}%", ha='center', va='bottom', fontsize=9)
    
    plt.axhline(avg_coverage, color='r', linestyle='--', linewidth=2,
                label=f'Average: {avg_coverage:.1f}%')
    plt.title(f"Word Coverage Analysis (Average: {avg_coverage:.1f}%)")
    plt.xlabel("Test Sentence")
    plt.ylabel("Coverage (%)")
    plt.xticks(range(len(sentences)), [f"S{i+1}" for i in range(len(sentences))], rotation=0)
    plt.ylim(0, 110)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

def generate_performance_plot(word_counts, gen_times, save_path):
    """Generate line chart showing video generation performance."""
    plt.figure(figsize=(10, 6))
    
    # Plot line with markers
    plt.plot(word_counts, gen_times, marker='o', linewidth=2, markersize=8, color='#2196F3')
    
    # Add value labels on points
    for x, y in zip(word_counts, gen_times):
        plt.text(x, y, f"{y:.2f}s", fontsize=10, ha='center', va='bottom')
    
    plt.title("Video Generation Performance", fontsize=16, fontweight='bold')
    plt.xlabel("Number of Words", fontsize=12)
    plt.ylabel("Generation Time (seconds)", fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Set background color
    ax = plt.gca()
    ax.set_facecolor('#f0f0f5')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

def generate_all_visualizations(video_dir=None, results_dir=None):
    """Generate all visualization plots."""
    results_dir = Path(results_dir) if results_dir else RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)
    
    print("=" * 60)
    print("📊 Generating Visualizations")
    print("=" * 60)
    
    # Get evaluation results
    print("\n1️⃣ Analyzing dataset coverage...")
    results = evaluate_dataset_coverage(video_dir)
    
    # Generate dataset composition plot
    print("2️⃣ Creating dataset composition chart...")
    generate_dataset_stats_plot(
        results['stats'],
        results_dir / "dataset_composition.png"
    )
    
    # Generate coverage plot
    print("3️⃣ Creating word coverage analysis chart...")
    generate_coverage_plot(
        results['tests'],
        results_dir / "word_coverage_analysis.png"
    )
    
    # Generate performance plot
    print("4️⃣ Creating video generation performance chart...")
    # Sample performance data (word count vs generation time)
    word_counts = [1, 4, 7, 8]
    gen_times = [0.29, 2.67, 8.77, 5.28]
    generate_performance_plot(
        word_counts,
        gen_times,
        results_dir / "video_generation_performance.png"
    )
    
    print(f"\n✅ All visualizations saved to: {results_dir}")
    print("   • dataset_composition.png")
    print("   • word_coverage_analysis.png")
    print("   • video_generation_performance.png")
    print("=" * 60)
    
    return results

# ===============================
# 🚀 MAIN FUNCTIONS
# ===============================
def setup_dataset():
    """Setup and validate dataset mappings."""
    print("=" * 60)
    print("ISL Dataset Setup")
    print("=" * 60)
    
    # Generate labels
    print("\n1️⃣ Generating labels.csv...")
    generate_labels_csv()
    
    # Validate videos
    print("\n2️⃣ Validating videos...")
    valid, invalid = validate_videos()
    print(f"✅ Valid videos: {valid}")
    if invalid:
        print(f"⚠️ Invalid videos: {len(invalid)}")
    
    # Load mapper
    print("\n3️⃣ Building video mappings...")
    mapper = ISLDatasetMapper()
    print(f"✅ Loaded {len(mapper.video_map)} video mappings")
    
    print("\n" + "=" * 60)
    print("✅ Dataset setup complete!")
    print("=" * 60)
    
    return mapper

if __name__ == "__main__":
    # Run full setup and evaluation
    mapper = setup_dataset()
    
    print("\n" + "=" * 60)
    print("Running evaluation...")
    print("=" * 60)
    
    results = evaluate_dataset_coverage()
    print_evaluation_report(results)
    
    print("\n" + "=" * 60)
    print("Generating visualizations...")
    print("=" * 60)
    
    generate_all_visualizations()
