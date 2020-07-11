def angleAddDeg(x,y):
    if (x > 0)    & (y >= 0): return 0
    elif (x < 0)  & (y <= 0): return 180
    elif (x < 0) & (y > 0):   return -180
    elif (x > 0) & (y < 0):  return -360
    elif (x == 0) & (y > 0):  return 90
    elif (x == 0) & (y > 0):  return 270
    else: return np.nan

def getAngleDeg(x,y):
    if (y == 0) | (x == 0): deg = 0
    else: deg = abs(np.degrees(np.arctan(y/x)))
    deg = abs(deg + angleAddDeg(x,y))
    return deg

def angleDiff(x1,y1,x2,y2):
    vec = np.array([getAngleDeg(x1,y1),getAngleDeg(x2,y2)])
    MAX, MIN = np.max(vec), np.min(vec)
    if (MAX - MIN) <= 180: return MAX - MIN
    else: return 360 - MAX + MIN
    
def rotate_angle_180(x):
    if x < 180: return x + 180
    else:       return  360 - (x + 180)
    
def absoluteAngle(angle):
    # map over angle
    if angle > 180: return 360 - angle
    else: return angle  
    