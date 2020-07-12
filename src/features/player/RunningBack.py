import pandas as pd
from src.numeric.NumericalMethods import NumericalMethods
 
class RunningBack(NumericalMethods):
    
    def __init__(self,verbose=False):

        self.verbose = verbose


    # get features relative to back
    def relativityTheory(self,df):
        
        # get back features
        carriers = self.backFeatures(df)

        # Get Info
        grpflds = ['GameId','PlayId','back_from_scrimmage','back_oriented_down_field','back_moving_down_field']
        calc    = ['X','Y','back_X','back_Y']
        player_distance = df[['GameId','PlayId','NflId','X','Y']]
        player_distance = pd.merge(player_distance, carriers, on=['GameId','PlayId'], how='inner')
        player_distance = player_distance[player_distance['NflId'] != player_distance['NflIdRusher']]
        player_distance['dist_to_back'] = player_distance[calc].apply(lambda x: self.euclideanDistance(x[0],x[1],x[2],x[3]), axis=1)
        player_distance = player_distance.groupby(grpflds).agg({'dist_to_back':['min','max','mean','std']}).reset_index()
        player_distance.columns = grpflds+ ['min_dist','max_dist','mean_dist','std_dist']
        return player_distance


    # get back features
    def backFeatures(self,df):
        carriers = df[df['NflId'] == df['NflIdRusher']][['GameId','PlayId','NflIdRusher','X','Y','Orientation','Dir','YardLine']]
        carriers['back_from_scrimmage'] = carriers['YardLine'] - carriers['X']
        carriers['back_oriented_down_field'] = carriers['Orientation'].apply(lambda x: self.backDirection(x))
        carriers['back_moving_down_field'] = carriers['Dir'].apply(lambda x: self.backDirection(x))
        carriers = carriers.rename(columns={'X':'back_X','Y':'back_Y'})
        return carriers[['GameId','PlayId','NflIdRusher','back_X','back_Y','back_from_scrimmage','back_oriented_down_field','back_moving_down_field']].copy()


    @staticmethod
    def backDirection(orientation):
        if orientation > 180.0: return 1
        else:                   return 0

