# Shakespearean Character Arc & Downfall Analyzer

An advanced AI/ML system to analyze emotional trajectories and psychological downfall patterns in Shakespeare characters using state-of-the-art NLP models.

## Overview

This project analyzes the complete emotional journey of **Lady Macbeth** (from *Macbeth*) and **Ophelia** (from *Hamlet*) by:

- **Emotion Detection**: Using transformer-based models fine-tuned for emotional analysis
- **Sentiment Analysis**: Tracking polarity and subjectivity across dialogues
- **Psychological Metrics**: Calculating distress scores and mental stability
- **Downfall Curves**: Generating visual representations of character decline

## Features

### 1. Multi-Model Emotion Detection
- **Primary Model**: `j-hartmann/emotion-english-distilroberta-base` - specialized for emotion classification
- **Sentiment Model**: `distilbert-base-uncased-finetuned-sst-2-english` - for sentiment polarity
- **TextBlob**: For additional linguistic analysis (polarity & subjectivity)

### 2. Comprehensive Emotion Tracking
Tracks 6 core emotions:
- Anger
- Fear
- Sadness
- Joy
- Disgust
- Surprise

### 3. Psychological Metrics
- **Distress Score**: Sum of negative emotions (anger, fear, sadness, disgust)
- **Mental Stability**: Inverse of distress (1 - distress_score)
- **Downfall Curve**: Weighted combination of:
  - 30% Mental Stability
  - 30% Sentiment Polarity
  - 20% Inverse Distress
  - 20% Joy

### 4. Visualizations
- Main downfall trajectory with act markers
- Emotion intensity heatmap over time
- Psychological metrics (stability vs distress)
- Sentiment polarity bar chart with moving average
- Overall emotion distribution pie chart
- Side-by-side character comparison

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 4GB+ RAM (for transformer models)

### Setup

1. **Install dependencies**:
```bash
pip install -r requirements.txt