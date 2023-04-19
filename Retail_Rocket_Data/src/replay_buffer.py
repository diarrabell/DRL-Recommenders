import os
import pandas as pd
import tensorflow as tf
from utility_new import to_pickled_df, pad_history
from tqdm import tqdm

class ReplayBuffer:
    def __init__(self, data_directory):
        self.data_directory = data_directory

    def create(self):
        print('Started to create the replay buffer')
        # session length
        length = 10

        sorted_events=pd.read_pickle(os.path.join(self.data_directory, 'sorted_events.df'))
        item_ids=sorted_events.item_id.unique()
        pad_item=len(item_ids)

        train_sessions = pd.read_pickle(os.path.join(self.data_directory, 'sampled_train.df'))
        groups=train_sessions.groupby('session_id')
        ids=train_sessions.session_id.unique()

        state, len_state, action, is_buy, next_state, len_next_state, is_done = [], [], [], [], [],[],[]

        for id in tqdm(ids):
            group=groups.get_group(id)
            history=[]
            group_dict = group.to_dict('index')
            for index, row in group_dict.items():
                s=list(history)
                len_state.append(length if len(s)>=length else 1 if len(s)==0 else len(s))
                s=pad_history(s,length,pad_item)
                a=row['item_id']
                is_b=row['is_buy']
                state.append(s)
                action.append(a)
                is_buy.append(is_b)
                history.append(row['item_id'])
                next_s=list(history)
                len_next_state.append(length if len(next_s)>=length else 1 if len(next_s)==0 else len(next_s))
                next_s=pad_history(next_s,length,pad_item)
                next_state.append(next_s)
                is_done.append(False)
            is_done[-1]=True

        dic={'state':state,'len_state':len_state,'action':action,'is_buy':is_buy,'next_state':next_state,'len_next_states':len_next_state,
            'is_done':is_done}
        replay_buffer=pd.DataFrame(data=dic)
        to_pickled_df(self.data_directory, replay_buffer=replay_buffer)

        dic={'state_size':[length],'item_num':[pad_item]}
        data_statis=pd.DataFrame(data=dic)
        print(data_statis)
        to_pickled_df(self.data_directory,data_statis=data_statis)
        print('Finished creating the replay buffer')