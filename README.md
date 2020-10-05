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

<img src="https://latex.codecogs.com/svg.latex?\Large&space;W^*=argmaxP(X|W)P(W)" title=" W ^*=arg max P(X|W) P(W)" />
where P(X|W) is the acoustic model and P(W) is the language model. 

In ASR, we can use a pronunciation table to produce the phones for the text sequence. Next, an acoustic model is needed for these phones.

## Feature extractors

1. MFCC
2. Raw spectograms (or log spectogram)
3. Mel power spectogram
4. PLP


For NN based models generally spectograms are used

## Models 

### 1. **GMM-HMM: Gaussian-Mixture-Model & Hidden-Markov-Model**


In this setup, the distribution of features of a phone is modelled by GMM(acoustic model). Given a phone, it models observed probability distribution of the feature vector. HMM models transition between phones 
and corresponding audio features (MFCC for example). HMM models how hidden states (phones) is transited and observed, i.e. what phones follow each other.


When a HMM model is learned, **forward learning** is used to calculate the likelihood of observations:

<img src="https://latex.codecogs.com/svg.latex?\Large&space;P(X)=\sum_SP(X,S)=\sum_SP(X|S)P(S)" title="\Sum_S P(X,S) = \Sum_S P(X|S) P(S)" />


where **P(X)** is probability of observed event, **P(X|S)** is probability of an observation given an internal state known as emission densition, **P(S)** is transition probability between 
internal states known as transition density. The likelihood of feature given phone is calculated by GMM: 

<img src="https://latex.codecogs.com/svg.latex?\Large&space;P(X|S)=\sum_jp_j\mathcal{N}(\mu_j,\sigma_j)" title="P(X|S)=\sum_jp_j\mathcalN(\mu_j,\sigma_j)" />.


We can learn the Gaussian model (p) for each phone from the training data.

To avoid exponential complexity in HMM modelling, probability up to some sequence is calculated instead of whole sequence. This learning of transition probability and conditional probability is done via algorithms such as **forward-backward** algorithm.

### 2. **DNN-HMM: Deep Neural Network & Hidden-Markov Model**

In this model, DNN is trained to provide posterior probability estimates for the HMM states (**P(S|X)**). 
Specifically, for an observation, the output of the DNN (**y(S)**) for the HMM state S is obtained.

<img src="https://latex.codecogs.com/svg.latex?\Large&space;y(S)=P(S|X)=\frac{exp(a(S))}{\sum_S'exp(a(S'))}" title="y(S)=P(S|X)=\frac{exp{a(S)}}{\sum_S'exp{a(S')}}" />.

where **a(S)** is the activation at the output layer corresponding
to state S. The recognizer uses a pseudo log-likelihood of state
S given observation X:

<img src="https://latex.codecogs.com/svg.latex?\Large&space;P(X|S)=logy(S)-logP(S)" title="P(X|S)=\logy(S)-\logP(S)" />.

where **P(S)** is the prior probability of state s calculated from the training data.



