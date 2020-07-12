from src.numeric.NumericalMethods import NumericalMethods 
from src.numeric.NumericAngle import NumericAngle 
from src.features.physics.ClassicalMechanics import ClassicalMechanics 
import numpy as np, pandas as pd, scipy as sp


class Defense(ClassicalMechanics,NumericalMethods,NumericAngle):

    def __init__(self):

        self.setTimeDelta()
    

    def forecast(self,df):

        # Create Definsive Dataset
        # ---------------------------------
    
        # get rusher
        meta_data   = ['PlayDirection','Season','YardLine']
        rush_fields = ['GameId','PlayId','RusherTeam','RusherX','RusherY','RushS','RushA','RushDir','RushOrient']
        rusher = df.loc[df['NflId'] == df['NflIdRusher']][['GameId','PlayId','Team','X','YNEW','S','A','DirII','OrientationII']].copy()
        rusher.columns = rush_fields
    
        # create table
        DD = pd.merge(df,rusher,on=['GameId','PlayId'],how='inner')
        DD = DD.loc[DD['Team'] != DD['RusherTeam']][rush_fields + ['X','YNEW','OrientationII','DirII','S','A'] + meta_data]
    
        # Calculate new future points
        DD['RusherFXV'] = self.physicsX(DD.RusherX,DD.RushS,DD.RushA,DD.RushDir)
        DD['RusherFYV'] = self.physicsY(DD.RusherY,DD.RushS,DD.RushA,DD.RushDir)
        DD['RusherOXV'] = self.physicsX(DD.RusherX,DD.RushS,DD.RushA,DD.RushOrient)
        DD['RusherOYV'] = self.physicsY(DD.RusherY,DD.RushS,DD.RushA,DD.RushOrient)
    
        # measure rushers distance from yard line & map Defenders Acceleration
        DD['RushDistYardLine'] = DD['RusherFXV'] - DD['YardLine']
    
        # calculate new defensive lineman points
        DD['FXV'] = self.physicsX(DD.X,DD.S,DD.A,DD.DirII)
        DD['FYV'] = self.physicsY(DD.YNEW,DD.S,DD.A,DD.DirII)

        DD['RusherFXV'] = self.physicsX(DD.RusherX,DD.RushS,DD.RushA,DD.RushDir)
        DD['RusherFYV'] = self.physicsY(DD.RusherY,DD.RushS,DD.RushA,DD.RushDir)
    
    
        # calculate rusher velocity
        DD['RusherVelocityX'] = self.physicsVx(DD.RushS,DD.RushA,DD.RushDir)
        DD['RusherVelocityY'] = self.physicsVy(DD.RushS,DD.RushA,DD.RushDir)
    
        # calculate the defensive man's velocity
        DD['VelocityX'] = self.physicsVx(DD.S,DD.A,DD.DirII)
        DD['VelocityY'] = self.physicsVy(DD.S,DD.A,DD.DirII)
    
        # calculate change in velocities
        DD['VelocityDeltaX'] = DD['VelocityX'] - DD['RusherVelocityX']
        DD['VelocityDeltaY'] = DD['VelocityY'] - DD['RusherVelocityY']
        DD['VelocityDelta']  = np.sqrt(DD['VelocityDeltaX']**2 + DD['VelocityDeltaY']**2)
    
        # Concept Features
        # ---------------------------

        # get new distance equations
        DD['StartDistance']   = self.euclideanDistance(DD['X'],DD['YNEW'],DD['RusherX'],DD['RusherY'])
        DD['RunnerHeaded']    = self.euclideanDistance(DD['FXV'],DD['FYV'],DD['RusherFXV'],DD['RusherFYV'])
        DD['RunnerThinking']  = self.euclideanDistance(DD['FXV'],DD['FYV'],DD['RusherOXV'],DD['RusherOYV'])
    
        # Calculate Time Till Impact
        # ---------------------------
        DD['TimeTillImpact']   = DD['VelocityDelta'] / (DD['RunnerHeaded'] + .01)
        DD['DistanceEstimate'] = DD['TimeTillImpact']*DD['RusherVelocityX'] + DD['YardLine'] + DD['RusherFXV'] - DD['RusherX']
    
        # Defender Assessing
        # ---------------------------

        # Defensive Frame, looking at Position
        DD['RushStartNormX'] = (DD['RusherX'] - DD['X'])/DD['StartDistance']
        DD['RushStartNormY'] = (DD['RusherY'] - DD['YNEW'])/DD['StartDistance']

        # Defensive Frame, Defender Assessing
        DD['DenfenderLookingX'] = np.cos(np.radians(DD['OrientationII']))
        DD['DefenderLookingY']  = np.sin(np.radians(DD['OrientationII']))

        # Rusher Assessing
        # ---------------------------

        # Rusher Frame, Rusher Position
        DD['DefStartNormX'] = (DD['X']    - DD['RusherX'])/DD['StartDistance']
        DD['DefStartNormY'] = (DD['YNEW'] - DD['RusherY'])/DD['StartDistance']

        # Rusher Frame, Rusher Assessing
        DD['RusherLookingX'] = np.cos(np.radians(DD['RushOrient']))
        DD['RusherLookingY'] = np.sin(np.radians(DD['RushOrient']))
    
        # Defender Engagement
        # ---------------------------

        # Calculate Normalized V Units
        DD['RushGoingNormX'] = (DD['RusherFXV'] - DD['X'])/DD['StartDistance']
        DD['RushGoingNormY'] = (DD['RusherFYV'] - DD['YNEW'])/DD['StartDistance']

        # Calculate Normalized Vector
        DD['DenfenderMovingX'] = np.cos(np.radians(DD['DirII']))
        DD['DefenderMovingY']  = np.sin(np.radians(DD['DirII']))


        # Engagement Orientation
        # --------------------------

        # Get Orientation Angle
        DD['DefenderEngagement'] = DD[['RushGoingNormX','RushGoingNormY','DenfenderMovingX','DefenderMovingY']].apply(lambda x: self.angleDiff(x[0],x[1],x[2],x[3]),axis=1)
        DD['DefenderAssessing']  = DD[['RushStartNormX','RushStartNormY','DenfenderLookingX','DefenderLookingY']].apply(lambda x: self.angleDiff(x[0],x[1],x[2],x[3]),axis=1)
        DD['RusherAssessing']    = DD[['DefStartNormX','DefStartNormY','RusherLookingX','RusherLookingY']].apply(lambda x: self.angleDiff(x[0],x[1],x[2],x[3]),axis=1)
        DD['CloseDistance']      = DD['RunnerHeaded'] - DD['StartDistance']

        # get non future location info
        # -----------------------------
        DD['XMAP'] = DD['X']      - DD['RusherX']
        DD['YMAP'] = DD['YNEW']   - DD['RusherY']
        DD['SMAP'] = abs(DD['S']) - abs(DD['RushS'])
    
    
        # setup dataframe
        full = pd.DataFrame()

        # map defensive players to table
        for name, group in DD.groupby(['PlayId','GameId']):
        
            # create variables
            info = {'RushDistYardLine': group['RushDistYardLine'].values[0],
                    'PlayId':           group['PlayId'].values[0],
                    'GameId':           group['GameId'].values[0]}
        
            x = 0
            for index, row in group.sort_values('RunnerHeaded').iterrows():
                x = x + 1
                for key in ['SMAP','CloseDistance','DefenderAssessing','RusherAssessing','RunnerHeaded','RunnerThinking','DefenderEngagement','DistanceEstimate','TimeTillImpact']:
                    if key+str(x) not in info.keys():
                        info[key+str(x)] = [DD.ix[index,key]]

            # concat full dataframe
            full = pd.concat([full,pd.DataFrame(info)],sort=False).reset_index(drop=True)
        return full
    

    # get defenseive state
    def Superposition(self,df):
    
        # Create Rusher Table
        # ------------------------
    
        # Field To Create Rush Table
        meta_data   = ['GameId','PlayId','Team','Xpoint','Ypoint','S','A','DirII','OrientationII']
        rush_fields = ['GameId','PlayId','RusherTeam','XpointRush','YpointRush','RushS','RushA','RushDir','RushOrient']
        add_fields  = ['X','YNEW','NflIdRusher','NflId']


        # X Point
        df['Xpoint'] = self.physicsX(df.X,df.S,df.A,df.DirII)
        df['Ypoint'] = self.physicsY(df.YNEW,df.S,df.A,df.DirII)

        # Fill Bad Points
        df['Xpoint'] = np.where(df['Xpoint'].isna(), df['X'],    df['Xpoint'])
        df['Ypoint'] = np.where(df['Ypoint'].isna(), df['YNEW'], df['Ypoint'])

        # get rusher
        rusher = df[df['NflId'] == df['NflIdRusher']][meta_data]
        rusher.columns = rush_fields
        rusher['Score'] = 1
    
    
        # Create Defensive Table
        # ------------------------

        # X Point
        df['Xpoint'] = self.physicsX(df.X,df.S,df.A,df.DirII,.6)
        df['Ypoint'] = self.physicsY(df.YNEW,df.S,df.A,df.DirII,.6)

        # Fill Bad Points
        df['Xpoint'] = np.where(df['Xpoint'].isna(), df['X'],    df['Xpoint'])
        df['Ypoint'] = np.where(df['Ypoint'].isna(), df['YNEW'], df['Ypoint'])

        # get rusher
        rusher = df[df['NflId'] == df['NflIdRusher']][meta_data]
        rusher.columns = rush_fields
        rusher['Score'] = 1

        # create table
        data = pd.merge(df[meta_data+add_fields],rusher,on=['GameId','PlayId'],how='inner')
        data['Score'] = data['Score'].fillna(-1)
        data.loc[data['Team'] != data['RusherTeam'],'Score'] = 1

        # create distance stats
        data['DistanceToRusher'] = np.sqrt((data['Xpoint'] - data['XpointRush'])**2 + (data['Ypoint'] - data['YpointRush'])**2)
        data["DefenseIndex"]     = data.groupby(['GameId','PlayId'])["DistanceToRusher"].rank("dense", ascending=True).astype(np.int)


        # Fill Bad Points (this may have created bad data*)
        data['X']    = np.where(data['NflId'] == data['NflIdRusher'],data['Xpoint'], data['X'])
        data['YNEW'] = np.where(data['NflId'] == data['NflIdRusher'],data['Ypoint'], data['YNEW'])

        # create data table
        data = data[['GameId','PlayId','X','YNEW','DistanceToRusher','DefenseIndex','Score','S']]


        # Compile Feature Set
        # -------------------------
    
        # create full
        full      = rusher[['GameId','PlayId']]
        meta_data = ['GameId','PlayId','X','YNEW']
        def_data  = ['GameId','PlayId','DefenderX','DefenderY']


        for key in np.linspace(1,2,2):
            Defender = data[data['DefenseIndex'] == key][meta_data]
            Defender.columns = def_data

            DST = data.merge(Defender,on    = ['GameId','PlayId'],how='left')
            DST['DistanceToRusher']         = np.sqrt((DST['X'] - DST['DefenderX'])**2 + (DST['YNEW'] - DST['DefenderY'])**2)
            DST['DefensiveInfluence'+str(int(key))] = self.sigmoid(DST['DistanceToRusher'],1,3)*DST['Score'] 
            DST  = DST.groupby(['GameId','PlayId'])[['DefensiveInfluence'+str(int(key))]].mean().reset_index()
            full = pd.merge(full,DST,on = ['GameId','PlayId'],how = 'left')
        
        # Model Input II Features
        feats = ['DefensiveInfluence1', 'DefensiveInfluence2']

        full['defenderMax']   = full[feats].max(axis=1)
        full['defenderMin']   = full[feats].min(axis=1)
        full['defenderSigma'] = full[feats].std(axis=1)    
    
        return full

    def defenseFeatures(self,df):

        calc = ['X','Y','RusherX','RusherY']
        rusher = df[df['NflId'] == df['NflIdRusher']][['GameId','PlayId','Team','X','Y']]
        rusher.columns = ['GameId','PlayId','RusherTeam','RusherX','RusherY']
        defense = pd.merge(df,rusher,on=['GameId','PlayId'],how='inner')
        defense = defense[defense['Team'] != defense['RusherTeam']][['GameId','PlayId','X','Y','RusherX','RusherY']]
        defense['def_dist_to_back'] = defense[calc].apply(lambda x: self.euclideanDistance(x[0],x[1],x[2],x[3]), axis=1)
        defense = defense.groupby(['GameId','PlayId']).agg({'def_dist_to_back':['min','max','mean','std']}).reset_index()
        defense.columns = ['GameId','PlayId','def_min_dist','def_max_dist','def_mean_dist','def_std_dist']
        return defense