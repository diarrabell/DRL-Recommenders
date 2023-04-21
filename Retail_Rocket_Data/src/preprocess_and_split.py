import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from utility_new import to_pickled_df

class DataPreparation:
    def __init__(self, data_directory):
        self.data_directory = data_directory

    def preprocess(self, item_properties_df):
        event_df = pd.read_csv(os.path.join(self.data_directory, 'events.csv'), header=0)
        event_df.columns = ['timestamp','session_id','behavior','item_id','transid']
        ###remove transid column
        event_df =event_df[event_df['transid'].isnull()]
        event_df = event_df.drop('transid',axis=1)
        ##########remove users with <=2 interactions
        event_df['valid_session'] = event_df.session_id.map(event_df.groupby('session_id')['item_id'].size() > 2)
        event_df = event_df.loc[event_df.valid_session].drop('valid_session', axis=1)
        ##########remove items with <=2 interactions
        event_df['valid_item'] = event_df.item_id.map(event_df.groupby('item_id')['session_id'].size() > 2)
        event_df = event_df.loc[event_df.valid_item].drop('valid_item', axis=1)
        ###### filter for events that have corresponding item properties
        item_ids_from_properties = item_properties_df['itemid'].unique()
        event_df=event_df[event_df['item_id'].isin(item_ids_from_properties)]
        ####### retrieve item ids for filtering item properties        
        item_ids_from_events = event_df['item_id'].unique()
        ######## transform to ids
        item_encoder = LabelEncoder()
        session_encoder= LabelEncoder()
        behavior_encoder=LabelEncoder()
        event_df['item_id'] = item_encoder.fit_transform(event_df.item_id)
        event_df['session_id'] = session_encoder.fit_transform(event_df.session_id)
        event_df['behavior']=behavior_encoder.fit_transform(event_df.behavior)
        ###########sorted by user and timestamp
        event_df['is_buy']=1-event_df['behavior']
        event_df = event_df.drop('behavior', axis=1)
        sorted_events = event_df.sort_values(by=['session_id', 'timestamp'])

        sorted_events.to_csv(os.path.join(self.data_directory, 'sorted_events.csv'), index=None, header=True)

        to_pickled_df(self.data_directory, sorted_events=sorted_events)
        return item_encoder, item_ids_from_events

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
