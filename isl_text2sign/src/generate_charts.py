"""
Generate ISL Visualizations

Run this script to generate all visualization charts:
- Word Coverage Analysis
- Video Generation Performance  
- Dataset Composition

Charts are saved to: model_training/results/
"""

from isl_mapper import generate_all_visualizations

if __name__ == "__main__":
    print("\n🎨 ISL Visualization Generator\n")
    
    # Generate all charts
    results = generate_all_visualizations()
    
    print("\n📊 Summary:")
    print(f"   Total Signs: {results['stats']['total_signs']}")
    print(f"   Letters: {results['stats']['letters']}")
    print(f"   Words: {results['stats']['words']}")
    print(f"\n   Average Coverage: {sum(t['coverage'] for t in results['tests']) / len(results['tests']):.1f}%")
    print("\n✅ Done! Check the results/ folder for charts.")
