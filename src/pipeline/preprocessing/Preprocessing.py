

class Preprocessing(object):
    
    # init data
    def __init__(self):

        # set coefficents
        self.velocityCoeff     = 1.1320096503100632
        self.accelerationCoeff = 1.1210484653841495
        self.year              = 2017

    # transform data
    def transform(self,df):


        if (df.Season==self.year).sum() > 0:

            # transform speed and acceleration
            df.loc[df.Season==self.year,'S'] = df.loc[df.Season==self.year,'S']*self.velocityCoeff
            df.loc[df.Season==self.year,'A'] = df.loc[df.Season==self.year,'A']*self.accelerationCoeff
        

        # transform orientation
        df['Orientation'] = df.apply(lambda x: self.rotate_angle_90(x.Orientation,x.Season),axis=1)

        # return dataframe
        return df


    @staticmethod
    def rotate_angle_90(angle,year):
        # Rotate 90 deg based on year

        if year != 2017: return angle
        if angle > 270:  return 90 - 360 + angle
        else:            return angle + 90

    

