# Opcode Based Android Malware Detection Using Machine Learning

1. Goal
 * A bachelor thesis demonstrating the application of ML in malware detection. The aim to the project is to demonstrate the use of ML techniques and feature compression in malware classification.

2. Data Preprocessing
 *  Decompiled .apk files using baksmali to access opcode-level instructions from .dex files.
 *  Tokenized opcode sequences from disassembled smali code by grouping them into n-grams of different lengths.
 *  Applied hash bucket encoding to convert opcode n-gram strings into feature vectors.

3. Model Training
 * Trained a NN and tested the effectiveness on different variations of processed data.
 * Evaluated model performance with 5-fold cross-validation.
 * Selected the best-performing hyperparameters for the final model

4. Results
Feature hashing proved to still be effective despite massive dimensionality reduction. Analysis revealed the efficacy of proposed method.
Model performed best with n-grams of length 3 and 2048 features.

| Metric    | Averages |
|-----------|-------|
| Accuracy  | 0.91  |
| Precision | 0.93  |
| Recall    | 0.90  |
| F1-Score  | 0.91  |
| ROC AUC   | 0.97  |

## Model Used
 
[Download model](https://drive.google.com/file/d/1dLunZLyczFkG7OmQA5jRTrcA44NQVeA7/view?usp=drive_link)

## Usage

Needs Baksmali disassembler to work https://github.com/JesusFreke/smali.
Download the model provided above. 
To test the solution run app.py and provide a path to an .apk file to be evaluated.

## Datasets Used

https://www.unb.ca/cic/datasets/andmal2017.html
https://github.com/mstfknn/android-malware-sample-library

In training used 1000 APKs total with an even split between malicious and benign files.
