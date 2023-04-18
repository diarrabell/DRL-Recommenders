import pandas as pd
import os
from tqdm import tqdm

class Popularity:
    def __init__(self, data_directory):
        self.data_directory = data_directory

    def create(self):
        print('Started to create the popularity dictionary')
        replay_buffer_behavior = pd.read_pickle(os.path.join(self.data_directory, 'sorted_events.df'))
        total_actions=replay_buffer_behavior.shape[0]
        pop_dict={}
        replay_buffer_behavior_dict = replay_buffer_behavior.to_dict('index')
        for index, row in tqdm(replay_buffer_behavior_dict.items()):
            action=row['item_id']
            if action in pop_dict:
                pop_dict[action]+=1
            else:
                pop_dict[action]=1
        for key in pop_dict:
            pop_dict[key]=float(pop_dict[key])/float(total_actions)
        f = open(os.path.join(self.data_directory, 'pop_dict.txt'), 'w')
        f.write(str(pop_dict))
        f.close()
        print('Finished creating the popularity dictionary')