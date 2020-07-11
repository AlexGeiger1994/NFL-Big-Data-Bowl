def getTimeDelta():
    return .625

# update X variable
def physicsX(x,v,a,angle,dt=0):
    if dt == 0: dt = getTimeDelta()
    projx = np.cos(np.radians(angle))      # get projection on x axis
    return x + v*projx*dt + .5*a*projx*(dt**2) # calculate x' given the span of time dt

# update Y variale
def physicsY(y,v,a,angle,dt=0):
    if dt == 0: dt = getTimeDelta()
    projy = np.sin(np.radians(angle))
    return  y + v*projy*dt + .5*a*projy*(dt**2)

# update velocity
def physicsVx(v,a,angle,dt=0):
    if dt == 0: dt = getTimeDelta()
    projx = np.cos(np.radians(angle))   
    return v*projx + a*projx*dt

# update velocity
def physicsVy(v,a,angle,dt=0):
    if dt == 0: dt = getTimeDelta()
    projy = np.sin(np.radians(angle)) # get projection on y axis
    return v*projy + a*projy*dt

# generally used for plotting points
def PhysicsXY(x,y,v,angle):
    endy = y + v* np.sin(np.radians(angle))
    endx = x + v* np.cos(np.radians(angle))
    return [x,endx], [y,endy]

# generally used for plotting points
def PhysicsVecXY(v,angle):
    endy = v* np.sin(np.radians(angle))
    endx = v* np.cos(np.radians(angle))
    return [endx],[endy]