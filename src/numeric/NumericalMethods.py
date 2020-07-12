import numpy as np, pandas as pd
from scipy import special as special

class NumericalMethods(object):
    
    @staticmethod
    def sigmoid(x,a,b):

        return special.expit(a*(b-x))



    @staticmethod
    def euclideanDistance(x1,y1,x2,y2):
        return np.sqrt((x1-x2)**2 + (y1-y2)**2)

