#!/usr/bin/env python3
"""
Visualization module for character downfall curves.
Creates comprehensive visualizations of emotional trajectories.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10


class ArcVisualizer:
    """
    Creates visualizations for character emotional arcs.
    """
    
    def __init__(self, csv_path: str, character_name: str):
        self.character_name = character_name
        self.df = pd.read_csv(csv_path)
        
    def plot_comprehensive_analysis(self, save_path: str = None):
        """
        Create a comprehensive multi-panel visualization.
        """
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)
        
        # 1. Main Downfall Curve
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_downfall_curve(ax1)
        
        # 2. Emotion Distribution Over Time
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_emotion_heatmap(ax2)
        
        # 3. Psychological Metrics
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_psychological_metrics(ax3)
        
        # 4. Sentiment Polarity
        ax4 = fig.add_subplot(gs[2, 0])
        self._plot_sentiment_polarity(ax4)
        
        # 5. Emotion Pie Chart
        ax5 = fig.add_subplot(gs[2, 1])
        self._plot_emotion_distribution(ax5)
        
        plt.suptitle(f"{self.character_name}'s Emotional Journey & Downfall Arc", 
                     fontsize=18, fontweight='bold', y=0.995)
        
        if save_path is None:
            save_path = f"/Users/nishh/Desktop/Research/results/{self.character_name}_complete_analysis.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
        plt.close()
        
    def _plot_downfall_curve(self, ax):
        """Plot the main downfall curve with act markers."""
        x = np.arange(len(self.df))
        y = self.df['downfall_curve']
        
        # Plot curve with gradient fill
        ax.plot(x, y, linewidth=3, color='#8B0000', label='Downfall Curve', zorder=3)
        ax.fill_between(x, y, alpha=0.3, color='#DC143C')
        
        # Mark acts with vertical lines
        acts = self.df['Act'].unique()
        act_changes = [0]
        for act in acts[:-1]:
            idx = self.df[self.df['Act'] == act].index[-1] + 1
            if idx < len(self.df):
                act_changes.append(idx)
                ax.axvline(x=idx, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
        
        # Add act labels
        for i, act in enumerate(acts):
            if i < len(act_changes):
                start = act_changes[i]
                end = act_changes[i+1] if i+1 < len(act_changes) else len(self.df)
                mid = (start + end) / 2
                ax.text(mid, ax.get_ylim()[1] * 0.95, f'Act {act}', 
                       ha='center', fontsize=11, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Calculate trend
        z = np.polyfit(x, y, 2)
        p = np.poly1d(z)
        ax.plot(x, p(x), "--", color='black', alpha=0.5, linewidth=2, label='Trend')
        
        ax.set_xlabel('Dialogue Progression', fontsize=12, fontweight='bold')
        ax.set_ylabel('Psychological Wellbeing Score', fontsize=12, fontweight='bold')
        ax.set_title('Character Downfall Trajectory', fontsize=14, fontweight='bold', pad=10)
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        
    def _plot_emotion_heatmap(self, ax):
        """Plot emotion intensity heatmap over time."""
        emotions = ['anger', 'fear', 'sadness', 'joy', 'disgust', 'surprise']
        emotion_data = self.df[emotions].T
        
        sns.heatmap(emotion_data, ax=ax, cmap='RdYlGn_r', cbar_kws={'label': 'Intensity'},
                    xticklabels=False, yticklabels=emotions, vmin=0, vmax=1)
        
        ax.set_xlabel('Dialogue Progression', fontsize=11, fontweight='bold')
        ax.set_ylabel('Emotions', fontsize=11, fontweight='bold')
        ax.set_title('Emotion Intensity Over Time', fontsize=12, fontweight='bold', pad=10)
        
    def _plot_psychological_metrics(self, ax):
        """Plot key psychological metrics."""
        x = np.arange(len(self.df))
        
        ax.plot(x, self.df['mental_stability'], label='Mental Stability', 
                linewidth=2, color='#2E8B57', alpha=0.8)
        ax.plot(x, self.df['distress_score'], label='Distress Score', 
                linewidth=2, color='#DC143C', alpha=0.8)
        
        ax.fill_between(x, self.df['mental_stability'], alpha=0.2, color='#2E8B57')
        ax.fill_between(x, self.df['distress_score'], alpha=0.2, color='#DC143C')
        
        ax.set_xlabel('Dialogue Progression', fontsize=11, fontweight='bold')
        ax.set_ylabel('Score', fontsize=11, fontweight='bold')
        ax.set_title('Psychological Metrics', fontsize=12, fontweight='bold', pad=10)
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        
    def _plot_sentiment_polarity(self, ax):
        """Plot sentiment polarity across dialogues."""
        x = np.arange(len(self.df))
        colors = ['green' if p > 0 else 'red' for p in self.df['polarity']]
        
        ax.bar(x, self.df['polarity'], color=colors, alpha=0.6, edgecolor='black', linewidth=0.5)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        # Add rolling average
        window = min(5, len(self.df))
        rolling_avg = self.df['polarity'].rolling(window=window, center=True).mean()
        ax.plot(x, rolling_avg, color='blue', linewidth=2.5, label='Moving Average', zorder=3)
        
        ax.set_xlabel('Dialogue Progression', fontsize=11, fontweight='bold')
        ax.set_ylabel('Sentiment Polarity', fontsize=11, fontweight='bold')
        ax.set_title('Sentiment Polarity Analysis', fontsize=12, fontweight='bold', pad=10)
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([-1, 1])
        
    def _plot_emotion_distribution(self, ax):
        """Plot overall emotion distribution."""
        emotions = ['anger', 'fear', 'sadness', 'joy', 'disgust', 'surprise']
        emotion_totals = self.df[emotions].sum()
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
        wedges, texts, autotexts = ax.pie(emotion_totals, labels=emotions, autopct='%1.1f%%',
                                           colors=colors, startangle=90, textprops={'fontsize': 10})
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            
        ax.set_title('Overall Emotion Distribution', fontsize=12, fontweight='bold', pad=10)
        
    def plot_comparison(self, other_visualizer, save_path: str = "comparison.png"):
        """
        Create a comparison plot between two characters.
        """
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        
        # Plot first character
        x1 = np.arange(len(self.df))
        y1 = self.df['downfall_curve']
        axes[0].plot(x1, y1, linewidth=3, color='#8B0000', label=self.character_name)
        axes[0].fill_between(x1, y1, alpha=0.3, color='#DC143C')
        axes[0].set_title(f"{self.character_name}'s Downfall Arc", fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Wellbeing Score', fontsize=11, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([0, 1])
        axes[0].legend()
        
        # Plot second character
        x2 = np.arange(len(other_visualizer.df))
        y2 = other_visualizer.df['downfall_curve']
        axes[1].plot(x2, y2, linewidth=3, color='#00008B', label=other_visualizer.character_name)
        axes[1].fill_between(x2, y2, alpha=0.3, color='#4169E1')
        axes[1].set_title(f"{other_visualizer.character_name}'s Downfall Arc", fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Dialogue Progression', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('Wellbeing Score', fontsize=11, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 1])
        axes[1].legend()
        
        plt.suptitle('Character Downfall Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison saved to {save_path}")
        plt.close()


def main():
    """
    Main visualization function.
    """
    print("="*70)
    print("   Character Arc Visualization")
    print("="*70)
    
    # Visualize Lady Macbeth
    print("\nGenerating Lady Macbeth visualizations...")
    lady_macbeth_viz = ArcVisualizer(
        csv_path='/Users/nishh/Desktop/Research/data/processed/LadyMacbeth_analysis_results.csv',
        character_name='Lady Macbeth'
    )
    lady_macbeth_viz.plot_comprehensive_analysis()
    
    # Visualize Ophelia
    print("\nGenerating Ophelia visualizations...")
    ophelia_viz = ArcVisualizer(
        csv_path='/Users/nishh/Desktop/Research/data/processed/Ophelia_analysis_results.csv',
        character_name='Ophelia'
    )
    ophelia_viz.plot_comprehensive_analysis()
    
    # Create comparison
    print("\nGenerating comparison visualization...")
    lady_macbeth_viz.plot_comparison(ophelia_viz, "LadyMacbeth_vs_Ophelia_comparison.png")
    
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  - Lady Macbeth_complete_analysis.png")
    print("  - Ophelia_complete_analysis.png")
    print("  - LadyMacbeth_vs_Ophelia_comparison.png")


if __name__ == "__main__":
    main()