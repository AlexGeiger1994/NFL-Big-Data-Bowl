import numpy as np
import pandas as pd

class ClassicalMechanics():
    
    def setTimeDelta(self):

        self.timeDelta = .625

    def getTimeDelta(self):

        return self.timeDelta


    
    def physicsX(self,x,v,a,angle,dt=0):

        # update X variable

        # if dt is zero get default time
        if dt == 0: dt = self.getTimeDelta()

        # get projection on x axis
        projx = np.cos(np.radians(angle)) 

        # calculate x' given the span of time dt

        return x + v*projx*dt + .5*a*projx*(dt**2)

    
    
    def physicsY(self,y,v,a,angle,dt=0):
        
        # update Y variale

        if dt == 0: dt = self.getTimeDelta()
        projy = np.sin(np.radians(angle))
        return  y + v*projy*dt + .5*a*projy*(dt**2)

    
    
    def physicsVx(self,v,a,angle,dt=0):

        # update velocity

        if dt == 0: dt = self.getTimeDelta()
        projx = np.cos(np.radians(angle))   
        return v*projx + a*projx*dt

    
    
    def physicsVy(self,v,a,angle,dt=0):

        # update velocity

        if dt == 0: dt = self.getTimeDelta()
        projy = np.sin(np.radians(angle)) # get projection on y axis
        return v*projy + a*projy*dt



    def PhysicsXY(self,x,y,v,angle):

        # generally used for plotting points

        endy = y + v* np.sin(np.radians(angle))
        endx = x + v* np.cos(np.radians(angle))
        return [x,endx], [y,endy]




