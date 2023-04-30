import argparse
import os
import subprocess
from preprocess_and_split import DataPreparation
from popularity import Popularity
from replay_buffer import ReplayBuffer
from item_features import ItemFeatures

def parse_args():
    parser = argparse.ArgumentParser(description='Set up for using SNQN recommender')

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

    item_features = ItemFeatures(data_directory)
    # Retrieve item properties
    item_properties_df = item_features.combine()

    data_preparation = DataPreparation(data_directory)
    # Created sorted events from dataset
    # Retrieve item encoder and IDs for creating item features
    item_encoder, item_ids_from_events = \
        data_preparation.preprocess(item_properties_df)
    # Split sorted events into training, validation, and test sessions
    data_preparation.split()

    # Create item features
    item_features.process(item_properties_df, item_encoder,
        item_ids_from_events)

    # Create popularity dictionary
    popularity = Popularity(data_directory)
    popularity.create()

    # Create replay buffer
    replay_buffer = ReplayBuffer(data_directory)
    replay_buffer.create()

if __name__ == '__main__':
    main()