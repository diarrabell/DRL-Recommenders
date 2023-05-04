import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from utility_new import to_pickled_df

class DataPreparation:
    def __init__(self, data_directory):
        self.data_directory = data_directory

    def preprocess(self, num_total_sessions, item_properties_df):
        print('Started to read and modify transactions_train')
        # Read file with column names in the first row
        all_transactions_df = pd.read_csv(os.path.join(self.data_directory, 'transactions_train.csv'), header=0)

        # Drop columns that aren't needed
        all_transactions_df = all_transactions_df.drop(columns=['price', 'sales_channel_id'])

        print("events df items:", len(all_transactions_df['article_id'].unique()))

        # Retrieve total number of unique items
        num_total_items = len(all_transactions_df['article_id'].unique())
        print(f'There are {num_total_items} total items from transactions_train')

        # Filter for events that have corresponding item properties
        item_ids_from_properties = item_properties_df['article_id'].unique()

        all_transactions_df=all_transactions_df[all_transactions_df['article_id'].isin(item_ids_from_properties)]

        item_ids_from_events = all_transactions_df['article_id'].unique()
        
        print("item properties items:", len(item_properties_df['article_id'].unique()))
       

        # Label encode 'article_id' (0 to N-1, where N is the number of total items)
        item_encoder = LabelEncoder()
        all_transactions_df['article_id'] = item_encoder.fit_transform(all_transactions_df.article_id)

        # Label encode 'customer_id' (0 to N-1, where N is the number of total customers)
        session_encoder = LabelEncoder()
        all_transactions_df['customer_id'] = session_encoder.fit_transform(all_transactions_df.customer_id)

        # Rename 'article_id' and 'customer_id' to match the field names in the code
        # Rename 't_dat' to 'date' to make column name more clear
        all_transactions_df = all_transactions_df.rename(columns={'article_id': 'item_id',
            'customer_id': 'session_id', 't_dat': 'date'})

        # Add 'is_buy' column to match the field name in the code and set all values as 1 for all transactions (purchases)
        all_transactions_df['is_buy'] = 1

        # Sort transactions by session_id, then by date, to create ordered transactions for all users
        sorted_transactions_df = all_transactions_df.sort_values(by=['session_id', 'date'])

        # Remove sessions (users) with less than three interactions to match the logic in the paper
        sorted_transactions_df['valid_session'] = sorted_transactions_df.session_id.map(
            sorted_transactions_df.groupby('session_id')['item_id'].size() > 2)
        sorted_transactions_df = sorted_transactions_df.loc[sorted_transactions_df.valid_session].drop('valid_session', axis=1)
        sorted_transactions_df = sorted_transactions_df.reset_index()

        # Filter for number of sessions (if needed)
        unique_session_ids = sorted_transactions_df['session_id'].unique()
        num_unique_session_ids = len(unique_session_ids)
        print(f'There are {num_unique_session_ids} total sessions after removing sessions with less than 3 interactions')
        if num_total_sessions < num_unique_session_ids:
            print(f'Started filtering for the input number of sessions, {num_total_sessions}')
            # Randomly choose num_total_sessions sessions from unique_session_ids
            chosen_session_ids = np.random.choice(
                unique_session_ids, num_total_sessions)
            sorted_transactions_df = sorted_transactions_df[
                sorted_transactions_df.session_id.isin(chosen_session_ids)]

        # Create file 'sorted_events.df'
        to_pickled_df(self.data_directory, sorted_events=sorted_transactions_df)
        print('Finished creating the sorted_events file')
        return num_total_items, item_encoder, item_ids_from_events

    def split(self):
        print('Started to split the sorted events into train, validation, and test')
        sorted_transactions_df = pd.read_pickle(os.path.join(self.data_directory, 'sorted_events.df'))

        total_sessions=sorted_transactions_df.session_id.unique()
        np.random.shuffle(total_sessions)

        fractions = np.array([0.8, 0.1, 0.1])
        # split into 3 parts
        train_ids, val_ids, test_ids = np.array_split(
            total_sessions, (fractions[:-1].cumsum() * len(total_sessions)).astype(int))

        train_sessions=sorted_transactions_df[sorted_transactions_df['session_id'].isin(train_ids)]
        val_sessions=sorted_transactions_df[sorted_transactions_df['session_id'].isin(val_ids)]
        test_sessions=sorted_transactions_df[sorted_transactions_df['session_id'].isin(test_ids)]

        to_pickled_df(self.data_directory, sampled_train=train_sessions)
        to_pickled_df(self.data_directory, sampled_val=val_sessions)
        to_pickled_df(self.data_directory,sampled_test=test_sessions)
        print('Finished splitting into train, validation, and test sessions')