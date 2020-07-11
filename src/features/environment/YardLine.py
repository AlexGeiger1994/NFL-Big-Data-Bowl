

class YardLine(object):

    # update yard line
    def getYardline(self,df):

        new_yardline = df[df['NflId'] == df['NflIdRusher']]
        new_yardline['YardLine'] = new_yardline[['PossessionTeam','FieldPosition','YardLine']].apply(lambda x: self.new_line(x[0],x[1],x[2]), axis=1)
        return new_yardline[['GameId','PlayId','YardLine']]


    @staticmethod
    def new_line(rush_team, field_position, yardline):

        if rush_team == field_position: return yardline + 0.0 
        else: return 100.0 - yardline # half the field plus the yards between midfield and the line of scrimmage
    