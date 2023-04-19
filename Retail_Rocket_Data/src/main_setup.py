import argparse
import os
import subprocess
from preprocess_and_split import DataPreparation
from popularity import Popularity
from replay_buffer import ReplayBuffer

def parse_args():
    parser = argparse.ArgumentParser(description='Set up before using SA2C')

    parser.add_argument('--data_directory', nargs='?',
        help='data directory for the Retail Rocket dataset')

    return parser.parse_args()

def main():
    args = parse_args()
    data_directory = args.data_directory

    # Download data to directory
    download_file_path = os.path.abspath(
      os.path.join(data_directory, '..', 'src/download.sh'))
    subprocess.call([download_file_path, data_directory])

    data_preparation = DataPreparation(data_directory)
    # Created sorted events from dataset
    data_preparation.preprocess()
    # Split sorted events into training, validation, and test sessions
    data_preparation.split()

    # Create popularity dictionary
    popularity = Popularity(data_directory)
    popularity.create()

    # Create replay buffer
    replay_buffer = ReplayBuffer(data_directory)
    replay_buffer.create()

if __name__ == '__main__':
    main()