# codingchallenge1020_team1

## October 1

1. Data Exploration:
- Skewed?
- How bad is the noise?
- How much empty space?


2. Data Preprocessing:
- band-pass filter?
- denoising?
- assume that more data will come


3. Data Augmentation?
- 


4. Models
- We have many classes with few samples (min. 3, max. 12)
- Random Forrest work well with little data
- GANs? hard
- 

## Automated Speech Recognition (ASR)

Primary objective of ASR is to predict text sequences (W) from a sequence of feature vectors (X)
Basic approach is to look for all possible sequences of words (with limited maximum length) and find one that matches the input acoustic features the best.

<img src="https://latex.codecogs.com/svg.latex?\Large&space;x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}" title="\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}" />

## Feature extractors

1. MFCC
2. Raw spectograms (or log spectogram)
3. Mel power spectogram
4. PLP
For NN based models generally spectograms are used

## Models 

1. GMM-HMM (before DL): Gaussian-Mixture-Model & Hidden-Markov-Model


