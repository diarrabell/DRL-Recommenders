#!/bin/bash
echo "Directory passed in for H&M data: $1"

mkdir ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

kaggle competitions download -c h-and-m-personalized-fashion-recommendations -f transactions_train.csv
unzip transactions_train.csv.zip
mv transactions_train.csv $1
echo "Finished downloading to $1"