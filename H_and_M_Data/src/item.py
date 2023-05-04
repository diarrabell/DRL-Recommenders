
import os
import numpy as np
import pandas as pd
import hashlib
from utility_new import to_pickled_df
from sklearn.preprocessing import StandardScaler, LabelEncoder


class ItemFeatures:
    def __init__(self, data_directory) -> None:
        self.data_directory = data_directory

    def create_df(self):
        # create dataframe for item properties
        item_properties_df = pd.read_csv(os.path.join(
            self.data_directory, 'articles.csv'), header=0)
        # drop 'detail_desc' column. multiple article_id can have the same product code
        item_properties_df = item_properties_df.drop(['detail_desc'], axis=1)
        return item_properties_df
    

    def create_features(self, item_properties_df, item_encoder, item_ids_from_events):
        # create item features
        #filter for items found in events df     
        item_properties_df = item_properties_df[
            item_properties_df['article_id'].isin(item_ids_from_events)] 
        # encode item ID      
        item_properties_df['article_id'] = item_encoder.transform(item_properties_df.article_id)
        #create label encoder and encode item properties 
        le = LabelEncoder()
        cols = item_properties_df.columns.values.tolist()
        #print(cols, type(cols))
        cols.remove('article_id')
        #print(item_properties_df[cols])
        item_properties_df = item_properties_df[cols].apply(le.fit_transform)
        to_pickled_df(self.data_directory, item_properties=item_properties_df)
        print("finished creating item features")




        
        
        
        

        
        

        



        

        









    
