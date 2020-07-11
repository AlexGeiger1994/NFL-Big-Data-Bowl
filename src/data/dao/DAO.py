import pandas as pd

class DAO(object):

    def __init__(self):

        self.path = 'C:/Users/Alex Geiger/Documents/GitHub/NFL-Big-Data-Bowl/'

    def getConnection(self):

        return 'data/train.csv'

    def getDataset(self):

        # set data types
        dtypes = {'WindSpeed': 'object'}

        return pd.read_csv(self.path + self.getConnection(), dtype=dtypes) 




