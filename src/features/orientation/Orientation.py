from src.features.environment.YardLine import YardLine 
import pandas as pd

class Orientation(YardLine):

    # update orientation
    def update(self,df):

        yardline = self.getYardline(df)

        df['YNEW']          = df[['Y','PlayDirection']].apply(lambda x: self.new_YNEW(x[0],x[1]), axis=1)
        df['X']             = df[['X','PlayDirection']].apply(lambda x: self.new_X(x[0],x[1]), axis=1)
        df['OrientationII'] = df[['Orientation','PlayDirection']].apply(lambda x: self.new_orientation_II(x[0],x[1]), axis=1)
        df['DirII']         = df[['Dir','PlayDirection']].apply(lambda x: self.new_orientation_II(x[0],x[1]), axis=1)
        df['Orientation']   = df[['Orientation','PlayDirection']].apply(lambda x: self.new_orientation(x[0],x[1]), axis=1)
        df['Dir']           = df[['Dir','PlayDirection']].apply(lambda x: self.new_orientation(x[0],x[1]), axis=1)
        df                  = df.drop('YardLine', axis=1)
        return pd.merge(df, yardline, on=['GameId','PlayId'], how='inner')


    @staticmethod
    def rotate_angle_90(angle,year):

        # Rotate 90 deg based on year

        if year != 2017: return angle
        if angle > 270:  return 90 - 360 + angle
        else:            return angle + 90

    @staticmethod
    def new_X(x_coordinate, play_direction):

        # flip x coordinates

        if play_direction == 'left': return 110.0 - x_coordinate
        else:                        return x_coordinate - 10.0

    @staticmethod
    def new_YNEW(y_coordinate, play_direction):

        # flip y coordinates

        if play_direction == 'left': return 53.3 - y_coordinate
        else:                        return y_coordinate

    @staticmethod
    def new_orientation(angle, play_direction):

        # Non-actual Angle Transpose

        if play_direction != 'left': return angle
        new_angle = 360.0 - angle
        if new_angle == 360.0: return 0.0
        return new_angle

    @staticmethod
    def new_orientation_II(angle, play_direction):

        # Actual Angle Transpose

        if (angle - 90) <= 0: angleII = 90 - angle
        else: angleII = 360 - angle + 90
        if play_direction == 'left': return (180+angleII) % 360
        else: return angleII
    
    



