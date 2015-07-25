import numpy as np
from functools import partial
import cv2
from cv2 import cv

xfile = "stl10/train_X.bin"
yfile = "stl10/train_y.bin"
imdim = 96
imsize = imdim * imdim
chan = 3

def get_label(off, yf=yfile):
    with open(yf, 'rb') as f2:
        f2.seek(off)
        n = f2.read(1)
        name = np.fromstring(n, dtype=np.uint8)
        nm =  name[0].astype(int)
        return str(nm)

def open_image(xf=xfile, show=False):
    with open(xf, 'rb') as f:
        count = 0
        lim = 2
        for chunk in iter(partial(f.read, imsize*chan), ''):
            img = np.zeros((imdim, imdim, chan), np.uint8)
            for e in range(chan):
                a = np.fromstring(chunk[e*imsize:(e+1)*imsize], dtype=np.uint8)
                im = a.reshape((imdim, imdim)).transpose()
                img[:,:,-(e+1)] = im
            if show:
                cv2.imshow(get_label(count), img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            count +=1
            if count >= lim:
                break


open_image(show=True)
