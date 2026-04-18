import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "detailed_metrics.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "model_comparison_chart.png")

def generate_matrix_dashboard():
    print("Generating Architectural Capability Matrix (Bar Chart Profile)...")
    
    # 1. Load Data
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
    else:
        print(f"Error: {CSV_PATH} not found.")
        return

    # 2. Data Preparation for Bar Chart
    df_melted = df.melt(
        id_vars="Model Architecture", 
        var_name="Evaluation Domain", 
        value_name="Accuracy Score"
    )
    
    # Standardizing naming for the legend
    df_melted["Evaluation Domain"] = df_melted["Evaluation Domain"].replace({
        "Training Accuracy": "Training Confidence",
        "Testing/Val Accuracy": "Testing/Validation Confidence"
    })

    # 3. Visualization Setup
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(15, 8))

    # Professional Premium Palette
    palette = {
        "Training Confidence": "#27ae60",  # Emerald
        "Testing/Validation Confidence": "#2980b9"   # Belize Hole
    }

    ax = sns.barplot(
        x="Model Architecture",
        y="Accuracy Score",
        hue="Evaluation Domain",
        data=df_melted,
        palette=palette,
        edgecolor=".2"
    )

    # 4. Annotation Logic
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(
                f"{height:.1%}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center", 
                va="center",
                fontsize=11, 
                color="#2c3e50",
                xytext=(0, 12), 
                textcoords="offset points",
                fontweight="bold"
            )

    # 5. Branding
    title_str = "Computer Vision Portfolio: Architectural Capability Matrix\n(Comparative Analysis of Training vs. Testing Performance)"
    plt.title(title_str, fontsize=18, fontweight='bold', pad=30, color='#2c3e50')
    plt.ylabel("Accuracy Performance Metric", fontsize=14, fontweight='demi', labelpad=15)
    plt.xlabel("Model Architecture Node (PyTorch/Keras)", fontsize=14, fontweight='demi', labelpad=15)
    
    plt.ylim(0, 1.15)
    plt.xticks(rotation=15, ha="right", fontsize=12)
    plt.yticks(np.arange(0, 1.1, 0.1), [f"{int(x*100)}%" for x in np.arange(0, 1.1, 0.1)], fontsize=11)
    
    plt.legend(
        title="Metric Metric Context", 
        bbox_to_anchor=(1.02, 1), 
        loc='upper left', 
        borderaxespad=0,
        fontsize=12,
        title_fontsize=13
    )

    sns.despine(trim=True)
    plt.tight_layout()
    
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
    print(f"Capability Matrix (Bar Chart) successfully exported to: {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_matrix_dashboard()
