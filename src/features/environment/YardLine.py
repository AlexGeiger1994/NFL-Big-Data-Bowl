

class YardLine(object):

    # update yard line
    def getYardline(self,df):

        # get yardline of rusher
        new_yardline = df.loc[df['NflId'] == df['NflIdRusher']].copy()

        # calculate new yardline
        new_yardline['YardLine'] = new_yardline[['PossessionTeam','FieldPosition','YardLine']].apply(lambda x: self.new_line(x[0],x[1],x[2]), axis=1)
        
        # return yardline info
        return new_yardline[['GameId','PlayId','YardLine']]


    @staticmethod
    def new_line(rush_team, field_position, yardline):

        # return yardline
        if rush_team == field_position: 
            return yardline
        
        # flip x-coordinate system
        return 100.0 - yardline
    