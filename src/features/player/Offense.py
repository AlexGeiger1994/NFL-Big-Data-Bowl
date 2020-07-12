
from src.numeric.NumericalMethods import NumericalMethods 
from src.features.physics.ClassicalMechanics import ClassicalMechanics 
import numpy as np, pandas as pd, scipy as sp


class Offense(ClassicalMechanics,NumericalMethods):
    
    def __init__(self,prefix='premier',alpha = 3,dt=[.2,.3,.4,.5,.6,.7,.8,.9,1.0]):

        self.dtDefault     = dt
        self.alphaDefault  = alpha
        self.prefixDefault = prefix
    
        # set time delta
        self.setTimeDelta()

    def forecast(self,df, prefix = '',alpha=0,addXY = True,dt_array = []):
        
        # set default parameters if not set
        if alpha == 0: alpha = self.alphaDefault
        if prefix == '': prefix = self.prefixDefault
        if len(dt_array) == 0: dt_array = self.dtDefault

        # get rusher
        meta_data   = ['PlayDirection','Season','YardLine']
        rush_fields = ['GameId','PlayId','RusherTeam','RusherX','RusherY','RushS','RushA','RushDir','RushOrient']
        rusher = df[df['NflId'] == df['NflIdRusher']][['GameId','PlayId','Team','X','YNEW','S','A','DirII','OrientationII']]
        rusher.columns = rush_fields

        # create table
        GK = pd.merge(df,rusher,on=['GameId','PlayId'],how='inner')
    
        # player score factors
        GK['PlayerFactor']                                         =  1
        GK.loc[GK['Team']     != GK['RusherTeam'],'PlayerFactor']  = -1
        GK.loc[GK['NflId']    == GK['NflIdRusher'],'PlayerFactor'] =  0
        GK.loc[GK['Position'] == 'QB',             'PlayerFactor'] =  0

        ids   = ['GameId','PlayId']
        new   = []
        rushx = []
        rushy = []
    
        for dt in dt_array:

            # get the field to be added
            added_field = prefix + 'PlayerInfluence' + str(int(dt*10000))
            added_rushx  = prefix + 'RusherX' + str(int(dt*10000))
            added_rushy  = prefix + 'RusherY' + str(int(dt*10000))
        
            # Calculate new future points
            GK[added_rushx] = self.physicsX(GK.RusherX,GK.RushS,GK.RushA,GK.RushDir,dt=dt)
            GK[added_rushy] = self.physicsY(GK.RusherY,GK.RushS,GK.RushA,GK.RushDir,dt=dt)

            # calculate new defensive lineman points
            GK['PlayerX'] = self.physicsX(GK.X,GK.S,GK.A,GK.DirII,dt=dt)
            GK['PlayerY'] = self.physicsY(GK.YNEW,GK.S,GK.A,GK.DirII,dt=dt)

            # player distance
            GK['PlayerDistance'] = self.euclideanDistance(GK[added_rushx],GK[added_rushy],GK.PlayerX,GK.PlayerY)
            GK[added_field]      = self.sigmoid(GK.PlayerDistance,1,alpha)*GK['PlayerFactor']
        
            # append new fields
            rushx.append(added_rushx)
            rushy.append(added_rushy)
            new.append(added_field)

        # take aggregate statistics
        data1 = GK[ids + new].groupby(['GameId','PlayId']).min().reset_index(drop=False)
        data2 = GK[ids + new].groupby(['GameId','PlayId']).max().reset_index(drop=False)
        data3 = GK[ids + new].groupby(['GameId','PlayId']).mean().reset_index(drop=False)
        data4 = GK[ids + new].groupby(['GameId','PlayId']).std().reset_index(drop=False)
        data5 = GK[ids + rushx].groupby(['GameId','PlayId']).mean().reset_index(drop=False)
        data6 = GK[ids + rushy].groupby(['GameId','PlayId']).mean().reset_index(drop=False)
    
        # change names
        data1.columns = ids + [s + 'Min'   for s in new]
        data2.columns = ids + [s + 'Max'   for s in new]
        data3.columns = ids + [s + 'Mu'    for s in new]
        data4.columns = ids + [s + 'Sigma' for s in new]
        data5.columns = ids + [s + 'RushX' for s in rushx]
        data6.columns = ids + [s + 'RushY' for s in rushy]
    
        # merge data
        data = pd.merge(data2,data1,how='left',on=['GameId','PlayId'])
        data = pd.merge(data,data3,how='left', on=['GameId','PlayId'])
        data = pd.merge(data,data4,how='left', on=['GameId','PlayId'])
    
        if addXY:
            data = pd.merge(data,data5,how='left', on=['GameId','PlayId'])
            data = pd.merge(data,data6,how='left', on=['GameId','PlayId'])
        
        return data




