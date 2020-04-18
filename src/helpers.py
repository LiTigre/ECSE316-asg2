import math
import numpy as np


def NextPowerOfTwo(number):
    # Returns next power of two following 'number'
    return math.ceil(math.log(number,2))

def pad_image(x):
    w, h = x.shape
    
    # width
    nextPower = NextPowerOfTwo(w)
    deficit_w = int(math.pow(2, nextPower) - w)
        
    # height
    nextPower = NextPowerOfTwo(h)
    deficit_h = int(math.pow(2, nextPower) - h)
    
    x = np.pad(x, ((0, deficit_w), (0, deficit_h)), mode='constant')
    
    return x
    