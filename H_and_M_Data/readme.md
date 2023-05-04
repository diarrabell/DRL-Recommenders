# H & M Dataset Project Structure

## Folder Description:
- data : folder to hold download kaggle data and subsequent generated files used for training
- src : contains edited source code for SNQN file and related files

## File Description:
The main file to train models on GPU is H&M_Recommenders.ipynb. This file downloads data from Kaggle and trains both models using GPU. 
The src folder contains the source code from the original DRL paper. New and edited files include:
- main.py : main function for preparing data and getting item features
- preprocess_and_split.py: contains class to organize, process and split transaction data
- item.py : contains a class that creates and encodes item features
- popularity.py : creates popularity dictionary
- SNQN_new.py : an optimized version of the original SNQN model 
- SNQN_with_features: a version of the original SNQN that uses GRU only (q learning disabled) and includes item features

## How to Run:
- Open H&M_Recommenders.ipynb
- Make sure runtime is set to GPU, preferably High RAM
- Run all cells. You will be asked for your Kaggle API access token. Upload your token to download kaggle data and complete training. 
