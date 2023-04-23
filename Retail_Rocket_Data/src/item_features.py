import os
import hashlib
import pandas as pd
from utility_new import to_pickled_df
from sklearn.preprocessing import StandardScaler

class ItemFeatures:
    def __init__(self, data_directory):
        self.data_directory = data_directory

    def combine(self):
        item_properties_p1_df = pd.read_csv(os.path.join(
            self.data_directory, 'item_properties_part1.csv'), header=0)
        item_properties_p2_df = pd.read_csv(os.path.join(
            self.data_directory, 'item_properties_part2.csv'), header=0)
        # Combine dataframes
        item_properties_df = pd.concat([item_properties_p1_df, item_properties_p2_df])
        # Get rid of the columns with 'available' or 'categoryid'
        item_properties_df = item_properties_df[
            (item_properties_df['property'] != 'available') &
            (item_properties_df['property'] != 'categoryid')
        ]
        return item_properties_df

    def process(self, item_properties_df, item_encoder,
        item_ids_from_events):
        print('Started to create item features')
        # Filter for items that are found in events
        item_properties_df = item_properties_df[
            item_properties_df['itemid'].isin(item_ids_from_events)]
        # Encode item ID
        item_properties_df['itemid'] = item_encoder.transform(
            item_properties_df.itemid)
        # Sort by timestamp
        item_properties_df = item_properties_df.sort_values(
            by=['timestamp'])
        # Filter for the items with the latest timestamps
        item_properties_df = item_properties_df.drop_duplicates(
            'itemid',keep='last')
        # Drop timestamp column
        item_properties_df = item_properties_df.drop(columns=['timestamp'])
        # Convert property column values to ints
        item_properties_df['property'] = item_properties_df['property'].astype(int)
        item_properties_df['value'] = item_properties_df['value'].apply(lambda x: int(hashlib.md5(x.encode()).hexdigest(), 16))
        # Standardize feature columns
        scaler = StandardScaler()
        columns_to_standardize = ['property', 'value']
        item_properties_df[columns_to_standardize] = \
            scaler.fit_transform(item_properties_df[columns_to_standardize])
        to_pickled_df(self.data_directory, item_properties=item_properties_df)
        print('Finished creating the item features')
