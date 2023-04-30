import argparse
import os
import subprocess
from preprocess_and_split import DataPreparation
from popularity import Popularity
from replay_buffer import ReplayBuffer

def parse_args():
    parser = argparse.ArgumentParser(description='Set up for using SNQN recommender')

    parser.add_argument('--data_directory', nargs='?',
        help='data directory for the H&M dataset')

    parser.add_argument('--num_total_sessions', nargs='?',
        help='number of total sessions, or customer IDs')
    return parser.parse_args()

def main():
    args = parse_args()
    data_directory = args.data_directory
    num_total_sessions = int(args.num_total_sessions)

    # Download data to directory
    download_file_path = os.path.abspath(
      os.path.join(data_directory, '..', 'src/download.sh'))
    subprocess.call([download_file_path, data_directory])

    data_preparation = DataPreparation(data_directory)
    # Created sorted events from dataset
    # Retrieve total number of items for statistics dictionary
    num_total_items = data_preparation.preprocess(num_total_sessions)
    # Split sorted events into training, validation, and test sessions
    data_preparation.split()

    # Create popularity dictionary
    popularity = Popularity(data_directory)
    popularity.create()

    # Create replay buffer
    replay_buffer = ReplayBuffer(data_directory)
    replay_buffer.create(num_total_items)

if __name__ == '__main__':
    main()