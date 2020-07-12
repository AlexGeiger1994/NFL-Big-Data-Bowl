class PositionMeta(object):
    
    # init count
   def __init__(self):

       self.positions = ['QB','CB','SS','NT','G','S','WR']
      

   # loop through positions
   def positionCount(self,df):
       
        for key in self.positions:

            self.countPositions(df,key)

   # count positions
   @staticmethod
   def countPositions(df,field):
        
        # count positions
        df[field + '_Count']  = 0
        df.loc[df['Position'] == field,field + '_Count'] = 1
        df[field + '_Count']  = df.groupby('PlayId')[field + '_Count'].transform('sum')