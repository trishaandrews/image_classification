import numpy as np
from functools import partial
import cv2
#from cv2 import cv

stldir = "stl10/"
xfile = stldir+"train_X.bin"
yfile = stldir+"train_y.bin"
ffile = stldir+"fold_indices.txt"
imdim = 96
imsize = imdim * imdim
chan = 3
tsh = 0.05

def show_image(img, count):
    #cv2.startWindowThread()
    cv2.imshow(get_label(count), img)
    #if (cv2.waitKey(1)&0xff) == 27:
    #    cv2.destroyAllWindows()
    cv2.waitKey(0)
    #if cv2.waitKey(0) & 0xff == 27:
    #    cv2.destroyAllWindows()
    #cv2.waitKey(1)
    cv2.destroyAllWindows()
    #cv2.waitKey(1)

def get_label(off, yf=yfile):
    with open(yf, 'rb') as f2:
        f2.seek(off)
        n = f2.read(1)
        name = np.fromstring(n, dtype=np.uint8)
        nm =  name[0].astype(int)
        return str(nm)

def sample_images(xf=xfile, show=False):
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
                show_image(img, count)
            count +=1
            if count >= lim:
                break

def open_image(off, xf=xfile):
    with open(xf, 'rb') as f:
        f.seek(off*imsize*chan)
        imgbs = f.read(imsize*chan)
        img = np.zeros((imdim, imdim, chan), np.uint8)
        for e in range(chan):
            a = np.fromstring(imgbs[e*imsize:(e+1)*imsize], dtype=np.uint8)
            im = a.reshape((imdim, imdim)).transpose()
            img[:,:,-(e+1)] = im
        return img

def make_gray(img):
    gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gimg

def get_folds(fi = ffile, lim=None):
    fis = []
    with open(fi, 'r') as inf:
        for line in inf:
            if lim:
                fis.append(np.random.choice(line.split(), lim))
            else:
                fis.append(line.split())
    folds = tuple(fis)
    return folds

def single_folds(classiftype, fi=ffile):
    fis = []
    with open(fi, 'r') as inf:
        for line in inf:
            flist = []
            for l in line.split():
                ind = int(l)
                #print l
                #print type(get_label(int(l)))
                #print "classiftype", int(classiftype)
                if get_label(ind) == str(classiftype):
                    flist.append(ind)
            fis.append(flist)
    folds = tuple(fis)
    return folds
            
def harris(gray):
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.04)
    
    #result is dilated for marking the corners, not important
    #dst = cv2.dilate(dst,None)
    
    # Threshold for an optimal value, it may vary depending on the image.
    ret, dst = cv2.threshold(dst,tsh*dst.max(),255,0)
    #threshold = tsh*dst.max()
    
    dst = np.uint8(dst)
    #gray [dst>threshold] = [0,0,255]
    #print dst, type(dst)
    #cv2.imshow('dst',dst)
    
    #if cv2.waitKey(0) & 0xff == 27:
    #    cv2.destroyAllWindows()
    return dst

#def kmeanscl(dat):
#    temp, classified_points, means = cv2.kmeans(data=dat, K=10, attempts=3, 
#flags=cv2.KMEANS_RANDOM_CENTERS) 
    
#def sift(gray):
#    sift = cv2.xfeatures2d.SIFT_create()
#    kp = sift.detect(gray,None)
#    #img=cv2.drawKeypoints(gray,kp,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#    img=cv2.drawKeypoints(gray,kp)
#    return img

print cv2.__version__
#sample_images(show=True)
#imagelist = [875]#, 4252, 410, 1471, 3904]
imagelist = single_folds(2)[0]
print imagelist
for ima in imagelist:
    img = open_image(ima)
    show_image(img, ima)
    img = make_gray(img)
    himg = harris(img)
    show_image(himg, ima)
    #simg = sift(img)
    #show_image(simg, ima)

#folds = get_folds(lim=3)
#print folds[0]

