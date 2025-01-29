---
sidebar_position: 1
title: Thesis
---

# Table of Contents

## List of Figures

## List of Tables

## List of Abbreviations

## Acknowledgements

## Title

**A Preliminary Study on Vowel Recognition using Convolutional Neural Networks for Individuals with Speech Disorders in the Italian Language**

## Abstract

Abstract: This study addresses the issue of speech and listening impairments, highlighting the potential for improvement through targeted training. The University of Padua's Computer engineering for music and multimedia (CSC) Lab is developing an online service, SoundRise, to help individuals refine their pronunciation skills. The platform provides a user-friendly interface for identifying vowel pitch and volume through audio analysis. As artificial intelligence continues to gain attention and improve various industries, this thesis aims to investigate the feasibility of integrating AI into this training service SoundRise. By incorporating AI training into vowel recognition, this thesis seeks to contribute to speech rehabilitation, offering a direct approach to learning and evaluating speech. This research explores vowel recognition in Italian using Convolutional Neural Networks (CNNs) and introduces a new dataset covering five vowels (/a/, /e/, /i/, /o/, and /u/) for future applications. The results demonstrate the effectiveness of this system in recognizing vowel types, potentially enhancing outcomes for individuals with speech disorders.

Italian Abstract: Questo studio affronta il problema delle difficoltà di parola e di ascolto, evidenziando il potenziale miglioramento attraverso un addestramento mirato. Il Laboratorio di Ingegneria Informatica per la Musica e i Multimedia (CSC) dell'Università di Padova sta sviluppando un servizio online, SoundRise, per aiutare le persone a perfezionare le proprie competenze di pronuncia. La piattaforma offre un'interfaccia intuitiva per identificare il tono e il volume delle vocali tramite analisi audio. Poiché l'intelligenza artificiale continua a suscitare interesse e a migliorare vari settori, questa tesi mira a indagare la fattibilità dell'integrazione dell'IA in questo servizio di addestramento, SoundRise. Integrando l'addestramento basato sull'IA nel riconoscimento delle vocali, la tesi si propone di contribuire alla riabilitazione del linguaggio, offrendo un approccio diretto all'apprendimento e alla valutazione del parlato. La ricerca esplora il riconoscimento delle vocali in italiano utilizzando reti neurali convoluzionali (CNN) e introduce un nuovo dataset che copre cinque vocali (/a/, /e/, /i/, /o/ e /u/) per applicazioni future. I risultati dimostrano l'efficacia di questo sistema nel riconoscimento dei tipi di vocali, con il potenziale di migliorare i risultati per le persone con disturbi del linguaggio.

## Keywords

CNN, Italian language, speech training, speech disorders, vowel recognition

## Introduction

Speech disorders are a significant health problem affecting millions of people worldwide. According to the World Health Organization, about 5% of children have speech development disorders [1]. Among them, congenital hearing impairment is one of the main causes of delayed speech development. These children often have difficulty developing language skills naturally because they cannot receive sound feedback normally.

Speech rehabilitation training is critical to the language development of these children. Research shows that early intervention and systematic speech training can significantly improve the language ability of children [2]. In speech training, vowel training is considered to be the most basic and critical link, because vowels are the basic unit of language and have an important impact on speech intelligibility [3].

Based on this background, this study develops a web-based speech training platform. This platform is an innovative upgrade based on the existing projects of the CSC Laboratory of the University of Padua. It mainly includes improvements in two aspects:

1. Optimize the user interface and interactive experience to make the platform more suitable for children
2. Introducing an intelligent speech recognition system based on convolutional neural network (CNN)

The main innovations of this study include:

- Proposed a sound information extraction method based on spectrogram
- Created a new dataset containing the five vowels of Italian, which contains multiple sound length samples from speakers of different genders
- Verified the effectiveness of the proposed dataset on the CNN model

## Related Works

The development of speech recognition technology has experienced an evolution from traditional methods to deep learning methods. Early research mainly relied on acoustic feature extraction and statistical models [4]. In recent years, deep learning, especially CNN, has made significant progress in the field of speech recognition [5]. However, through experimental comparison, it was found that in specific vowel recognition tasks, the traditional method still showed high accuracy and efficiency. This may be due to the relative simplicity of vowel features, which traditional methods have been able to capture well.

## Web application development



## Methodology

### System Architecture

// Add system architecture diagram and description

### Dataset

#### Data Collection

The dataset consists of Italian vowel audio samples (/a/, /e/, /i/, /o/, /u/):

- Total samples: 1000 audio files
- Training set: 800 samples (80%)
- Validation set: 200 samples (20%)
- Class distribution: Balanced across 5 vowel classes
- Audio generation: Used Google Text-to-Speech (gTTS) with various parameters
- Sample variations: Multiple pronunciations, durations, and intonations

#### Data Preprocessing

A. Audio Recording

- Used gTTS library for generating synthetic speech
- Multiple variations per vowel:
  - Standard pronunciations
  - Extended durations
  - Emphasized pronunciations
  - Combined sounds
  - Soft pronunciations
  - Tonal variations

B. Converting wave signal to image form

- Generated spectrograms from audio files
- Image size: 55x55 pixels
- Normalized pixel values to range [0,1]
- Applied data augmentation:
  - Width shift: ±10%
  - Height shift: ±10%
  - Zoom range: ±10%
  - Brightness variation: ±20%

### Model Architecture

#### CNN Structure

The final model uses a transfer learning approach with ResNet50V2:

1. Base Model:

- Pre-trained ResNet50V2 (weights='imagenet')
- Input shape: (55, 55, 3)
- Global average pooling
- Trainable parameters frozen initially

2. Classification Head:

- Dense layer (256 units, ReLU activation)
- Dropout (0.5)
- Output layer (5 units, Softmax activation)

#### Training Process

1. Training Strategy:

- Two-phase training approach:
  - Phase 1: Train only classification layers
  - Phase 2: Fine-tune last 10 layers of base model

2. Training Parameters:

- Batch size: 32
- Initial learning rate: 0.01
- Optimizer: SGD with momentum (0.9)
- Loss function: Categorical crossentropy
- Epochs: 50 (with early stopping)

3. Callbacks:

- Early stopping (patience=5, monitor='val_accuracy')
- Learning rate reduction (factor=0.2, patience=3)
- Model checkpoint (save best weights)

## Experimental Setup

### Environment

// Add experimental environment details

### Implementation Details

1. Development Environment:

- Python 3.x
- TensorFlow/Keras
- Libraries: gTTS, numpy, matplotlib

2. Data Processing Pipeline:

- Audio generation using gTTS
- Spectrogram conversion
- Data augmentation
- Train-validation split

3. Model Implementation:

- Transfer learning with ResNet50V2
- Custom training loops
- Callback implementations
- Performance monitoring

### Evaluation Metrics

// Add description of evaluation metrics

## Results

### Performance Analysis

The model achieved significant improvements through iterations:

1. Initial Results:

- Training accuracy: ~20%
- Validation accuracy: ~20%

2. After Transfer Learning:

- Training accuracy: 78.01%
- Validation accuracy: 85.50%
- Training loss: 0.4548
- Validation loss: 0.3831

3. Training Progression:

- Epoch 1: 52.50% validation accuracy
- Epoch 4: 76.50% validation accuracy
- Epoch 6: 81.00% validation accuracy
- Epoch 7: 85.50% validation accuracy

4. Key Observations:

- Consistent improvement in both training and validation metrics
- No significant overfitting (validation metrics better than training)
- Stable learning curve with steady improvements
- Effective transfer learning from ResNet50V2

### Comparative Study

// Add comparison with other methods

### Discussion

// Add result discussion

## Conclusion

- Main findings of the research
- Summary of innovations
- Limitations of the research
- Contributions to the field

## Future Work

In the future, the laboratory can further develop a speech-based AI training system based on this CNN model for speech training.

- Collect more audio data from real people for training and testing
- In image conversion, you can focus on the features of the spectrogram that are most related to vowels, such as the shape of the spectrogram, the energy distribution of the spectrogram, the frequency distribution of the spectrogram, etc.
- In model training, you can try to use more models, such as Transformer model, LSTM model, etc., to improve the performance of the model.
- In model evaluation, you can try to use more evaluation indicators, such as F1 score, accuracy, recall, precision, etc., to evaluate the performance of the model.

## References

[1] World Health Organization. (2021). World report on hearing.

[2] Johnson, C. J., & Beitchman, J. H. (2006). Language development and literacy: Early identification of language delay.

[3] Peterson, G. E., & Barney, H. L. (1952). Control methods used in a study of the vowels.

[4] Rabiner, L., & Juang, B. H. (1993). Fundamentals of speech recognition.

[5] Zhang, X., et al. (2017). Deep learning for speech recognition: A comprehensive review.

## Appendices

### Appendix A: Dataset Details

### Appendix B: Experimental Results

### Appendix C: Code Implementation

### Appendix D: Questionnaires and Protocols
