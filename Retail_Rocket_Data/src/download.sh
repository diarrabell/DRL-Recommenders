#!/bin/bash
echo "Directory passed in for Retail Rocket data: $1"

mkdir ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

kaggle datasets download -d retailrocket/ecommerce-dataset
unzip ecommerce-dataset.zip -d $1
echo "Finished downloading to $1"