# CSE260D Final Project: Perplexity and Diversity-Guided Submodular Selection for Stock Turning Points

**By:** Madison Killen, Siddhi Patel, Soroush Barmooz, and Soumya Kalluri

## Overview

This project constructs a highly informative dataset using advanced pruning methods (perplexity and diversity) combined with submodular selection to predict stock turning points. Using EURUSD currency pair movement data as a proxy for stock market patterns, we demonstrate that intelligent data pruning can create compact, highly predictive datasets while reducing computational requirements.

## Abstract

We developed a methodology to prune a raw dataset of 217k rows down to 2,000 samples while maintaining descriptive power for predicting stock turning points. By defining 20 target labels to classify directional movements and potential reversal zones (PRZs), we applied concepts from Marion et al. [1] and Ankner et al. [2] using the Lazy Greedy Algorithm for submodular optimization. We then ran multiple ablation studies and experiments to capture patterns of the pruned datasets and how it fairs through different model architectures.  

### Main Components

1. **Data Pruning Pipeline**
   - Perplexity calculation
   - Diversity measurement
   - Lazy Greedy Algorithm implementation for submodular selection

2. **Model Training**
   - ConvLSTM architecture for time-series prediction
   - Training on pruned datasets (2,000 samples)
   - Comparison with "golden" dataset (5,000 hand-pruned samples)

3. **Ablation Studies** (Figures 8-11)
   - Individual impact of perplexity pruning
   - Individual impact of diversity pruning
   - Combined effects analysis

4. **Large Scale Experimentation** (Figure 16)
   - Pruned dataset vs. "golden" dataset results
   - Heavy vs. light model comparisons

## Dataset

- **Source**: EURUSD currency pair movement data
- **Original size**: 217,000 rows
- **Pruned size**: 2,000 samples
- **Target labels**: 20 classes representing directional movements and potential reversal zones (PRZs)

## Methodology

1. Calculate perplexity and diversity metrics separately
2. Apply Lazy Greedy Algorithm for submodular selection
3. Train ConvLSTM on pruned dataset
4. Conduct ablation studies to evaluate individual pruning strategies
5. Compare against hand-pruned "golden" dataset baseline
6. Run large-scale experiments that accounts for full training data and heavy vs lightweight model architectures

## References

[1] Marion et al. - Referenced for perplexity-based pruning concepts  
[2] Ankner et al. - Referenced for diversity-based selection methods

## Link to Results
The [results](https://drive.google.com/drive/folders/1ASX5ANovrGU4VfSEcrMWb6zwTyh2kpl2?dmr=1&ec=wgc-drive-hero-goto) were too large to be pushed to git, so we included them in a Google Drive for reference. 
