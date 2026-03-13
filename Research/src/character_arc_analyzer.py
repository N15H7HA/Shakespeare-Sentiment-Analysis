#!/usr/bin/env python3
"""
Shakespearean Character Arc Analyzer
Analyzes emotional trajectories and downfall patterns in Shakespeare characters
using NLP, sentiment analysis, and emotion detection models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from textblob import TextBlob
import torch
import re
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class ShakespeareEmotionAnalyzer:
    """
    Analyzes emotions and psychological states in Shakespearean text.
    Uses multiple models for comprehensive emotion detection.
    """
    
    def __init__(self):
        print("Initializing emotion detection models...")
        
        # Load emotion detection model (fine-tuned for literary text)
        self.emotion_classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None
        )
        
        # Load sentiment analysis model
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
        
        print("Models loaded successfully!")
    
    def clean_shakespearean_text(self, text: str) -> str:
        """
        Clean and preprocess Shakespearean text while preserving meaning.
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove stage directions in brackets
        text = re.sub(r'\[.*?\]', '', text)
        return text.strip()
    
    def analyze_emotion(self, text: str) -> Dict:
        """
        Perform comprehensive emotion analysis on text.
        Returns emotions with scores and sentiment.
        """
        text = self.clean_shakespearean_text(text)
        
        if not text or len(text) < 3:
            return self._empty_result()
        
        # Get emotion scores
        try:
            emotion_results = self.emotion_classifier(text[:512])[0]  # Limit to 512 tokens
            emotions = {e['label']: e['score'] for e in emotion_results}
        except Exception as e:
            print(f"Emotion analysis error: {e}")
            emotions = {}
        
        # Get sentiment
        try:
            sentiment_result = self.sentiment_analyzer(text[:512])[0]
            sentiment = sentiment_result['label']
            sentiment_score = sentiment_result['score'] if sentiment == 'POSITIVE' else -sentiment_result['score']
        except Exception as e:
            print(f"Sentiment analysis error: {e}")
            sentiment = 'NEUTRAL'
            sentiment_score = 0.0
        
        # TextBlob for additional sentiment and subjectivity
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        return {
            'emotions': emotions,
            'sentiment': sentiment,
            'sentiment_score': sentiment_score,
            'polarity': polarity,
            'subjectivity': subjectivity,
            'text_length': len(text.split())
        }
    
    def _empty_result(self) -> Dict:
        """Return empty result structure."""
        return {
            'emotions': {},
            'sentiment': 'NEUTRAL',
            'sentiment_score': 0.0,
            'polarity': 0.0,
            'subjectivity': 0.0,
            'text_length': 0
        }


class CharacterArcAnalyzer:
    """
    Analyzes character arcs and tracks downfall patterns.
    """
    
    def __init__(self, csv_path: str, character_name: str):
        self.character_name = character_name
        self.df = pd.read_csv(csv_path)
        self.emotion_analyzer = ShakespeareEmotionAnalyzer()
        self.analysis_results = None
        
    def analyze_character_arc(self) -> pd.DataFrame:
        """
        Analyze the complete character arc through the play.
        """
        print(f"\nAnalyzing {self.character_name}'s journey...")
        
        results = []
        
        for idx, row in self.df.iterrows():
            print(f"Processing dialogue {idx + 1}/{len(self.df)}...", end='\r')
            
            dialogue = str(row['Dialogue'])
            analysis = self.emotion_analyzer.analyze_emotion(dialogue)
            
            # Extract dominant emotion
            emotions = analysis['emotions']
            dominant_emotion = max(emotions, key=emotions.get) if emotions else 'neutral'
            dominant_emotion_score = emotions.get(dominant_emotion, 0.0)
            
            # Calculate psychological distress indicators
            negative_emotions = ['anger', 'fear', 'sadness', 'disgust']
            distress_score = sum(emotions.get(e, 0.0) for e in negative_emotions)
            
            # Calculate mental stability (inverse of distress)
            mental_stability = 1.0 - distress_score
            
            result = {
                'Sno': row['Sno'],
                'Act': row['Act'],
                'Scene': row['Scene'],
                'Dialogue': dialogue[:100] + '...' if len(dialogue) > 100 else dialogue,
                'dominant_emotion': dominant_emotion,
                'dominant_emotion_score': dominant_emotion_score,
                'sentiment_score': analysis['sentiment_score'],
                'polarity': analysis['polarity'],
                'subjectivity': analysis['subjectivity'],
                'distress_score': distress_score,
                'mental_stability': mental_stability,
                'anger': emotions.get('anger', 0.0),
                'fear': emotions.get('fear', 0.0),
                'sadness': emotions.get('sadness', 0.0),
                'joy': emotions.get('joy', 0.0),
                'disgust': emotions.get('disgust', 0.0),
                'surprise': emotions.get('surprise', 0.0),
                'text_length': analysis['text_length']
            }
            
            results.append(result)
        
        print("\nAnalysis complete!")
        self.analysis_results = pd.DataFrame(results)
        return self.analysis_results
    
    def calculate_downfall_curve(self) -> np.ndarray:
        """
        Calculate the downfall curve based on multiple psychological metrics.
        Lower values indicate greater downfall.
        """
        if self.analysis_results is None:
            raise ValueError("Must run analyze_character_arc() first")
        
        # Weighted combination of metrics
        downfall_curve = (
            0.4 * self.analysis_results['mental_stability'] +  # Increase weight
            0.3 * (self.analysis_results['polarity'] + 1) / 2 +
            0.2 * (1 - self.analysis_results['distress_score']) +
            0.1 * self.analysis_results['joy']  # Decrease weight
        )
        
        # Apply moving average for smoothing
        window_size = min(3, len(downfall_curve))
        if window_size > 1:
            downfall_curve = np.convolve(downfall_curve, np.ones(window_size)/window_size, mode='same')
        
        return downfall_curve
    
    def save_results(self, output_path: str = None):
        """
        Save analysis results to CSV.
        """
        if self.analysis_results is None:
            raise ValueError("Must run analyze_character_arc() first")
        
        if output_path is None:
            output_path = f"/Users/nishh/Desktop/Research/data/processed/{self.character_name}_analysis_results.csv"
        
        # Add downfall curve to results
        results_with_curve = self.analysis_results.copy()
        results_with_curve['downfall_curve'] = self.calculate_downfall_curve()
        
        results_with_curve.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")
        
        return output_path


def main():
    """
    Main execution function.
    """
    print("="*70)
    print("   Shakespearean Character Arc & Downfall Analyzer")
    print("="*70)
    
    # Analyze Lady Macbeth
    print("\n" + "="*70)
    print("ANALYZING LADY MACBETH")
    print("="*70)
    
    lady_macbeth_analyzer = CharacterArcAnalyzer(
        csv_path='/Users/nishh/Desktop/Research/data/raw/LadyMacbeth.csv',
        character_name='LadyMacbeth'
    )
    lady_macbeth_results = lady_macbeth_analyzer.analyze_character_arc()
    lady_macbeth_output = lady_macbeth_analyzer.save_results()
    
    # Analyze Ophelia
    print("\n" + "="*70)
    print("ANALYZING OPHELIA")
    print("="*70)
    
    ophelia_analyzer = CharacterArcAnalyzer(
        csv_path='/Users/nishh/Desktop/Research/data/raw/Ophelia.csv',
        character_name='Ophelia'
    )
    ophelia_results = ophelia_analyzer.analyze_character_arc()
    ophelia_output = ophelia_analyzer.save_results()
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nResults saved:")
    print(f"  - {lady_macbeth_output}")
    print(f"  - {ophelia_output}")
    print("\nRun visualize_arcs.py to generate downfall curve visualizations.")


if __name__ == "__main__":
    main()