#import Image #as pil
import numpy as np
from functools import partial
import cv2
from cv2 import cv

infile = "train_X.bin"
imdim = 96
imsize = imdim * imdim
chan = 3

with open(infile, 'rb') as f:
    count = 0
    lim = 50
    for chunk in iter(partial(f.read, imsize*chan), ''):
        img = np.zeros((imdim, imdim, chan), np.uint8)
        for e in range(chan):
            a = np.fromstring(chunk[e*imsize:(e+1)*imsize], dtype=np.uint8)
            im = a.reshape((imdim, imdim)).transpose()
            img[:,:,-(e+1)] = im
        
        cv2.imshow("img", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        count +=1
        if count >= lim:
            break
