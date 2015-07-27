import numpy as np
from functools import partial
import cv2

class OpenImages:

    def __init__(self, stld="stl10/", xfl="train_X.bin", yfl="train_y.bin", 
                 ffl="fold_indices.txt", class_names="class_names.txt",
                 image_dimensions=96, channels=3, threshold=0.05):
        stldir = stld
        self.xfile = stldir+xfl
        self.yfile = stldir+yfl
        self.ffile = stldir+ffl
        self.classes = stldir+class_names
        self.IMDIM = image_dimensions
        self.IMSIZE = self.IMDIM * self.IMDIM
        self.CHAN = channels
        self.IMCHSIZE = self.IMSIZE * self.CHAN
        self.TSH = threshold

    def show_image(self, img, count):
        cv2.imshow(str(self.get_label(count)), img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def get_label(self, off):
        with open(self.yfile, 'rb') as f2:
            f2.seek(off)
            n = f2.read(1)
            name = np.fromstring(n, dtype=np.uint8)
            nm =  name[0].astype(int)
            return nm

    def get_class_names(self):
        count = 1
        class_names = []
        with open(self.classes, 'r') as cf:
            for line in cf:
                class_names.append((count, line.strip()))
                count += 1
        return class_names
            
    def sample_images(self, show=False, lim=10):
        with open(self.xfile, 'rb') as f:
            count = 0
            for chunk in iter(partial(f.read, self.IMCHSIZE), ''):
                img = np.zeros((self.IMDIM, self.IMDIM, self.CHAN), 
                               np.uint8)
                for e in range(self.CHAN):
                    a = np.fromstring(chunk[e*self.IMSIZE:(e+1)*
                                            self.IMSIZE], dtype=np.uint8)
                    im = a.reshape((self.IMDIM, self.IMDIM)).transpose()
                    img[:,:,-(e+1)] = im
                if show:
                    self.show_image(img, count)
                count +=1
                if count >= lim:
                    break

    def open_image(self, off):
        with open(self.xfile, 'rb') as f:
            f.seek(off*self.IMCHSIZE)
            imgbs = f.read(self.IMCHSIZE)
            img = np.zeros((self.IMDIM, self.IMDIM, self.CHAN), np.uint8)
            for e in range(self.CHAN):
                a = np.fromstring(imgbs[e*self.IMSIZE:(e+1)*self.IMSIZE], 
                                  dtype=np.uint8)
                im = a.reshape((self.IMDIM, self.IMDIM)).transpose()
                img[:,:,-(e+1)] = im
            return img

    def make_gray(self, img):
        gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gimg

    def get_folds(self, lim=None):
        fis = []
        with open(self.ffile, 'r') as inf:
            for line in inf:
                intlist = [int(l) for l in line.split()]
                if lim:
                    fis.append(list(np.random.choice(intlist, lim)))
                else:
                    fis.append(intlist)
        folds = tuple(fis)
        return folds

    def single_folds(self, classiftype, lim=None):
        fis = []
        with open(self.ffile, 'r') as inf:
            for line in inf:
                flist = []
                for l in line.split():
                    ind = int(l)
                    if self.get_label(ind) == classiftype:
                        flist.append(ind)
                    if lim:
                        if len(flist) >= lim:
                            break
                fis.append(flist)
        folds = tuple(fis)
        return folds
        
    def get_fold_images(self, lim=None):
        fold_data = []
        folds = oi.get_folds(lim)  #([f0],[f1],...)
        for i, f in enumerate(folds): #[f0],[f1],..
            fold_num = i+1
            for off in f: #f1 = [0, 400, 3845, 2, ...]
                fold_data.append((off, fold_num))
        return fold_data

    def get_all_labels(self, lim=None):
        with open(self.xfile, 'rb') as df:
            data = []
            index = 0
            for chunk in iter(partial(df.read, self.IMCHSIZE), ''):
                label = self.get_label(index)
                data.append((index, label))
                index += 1
                if lim:
                    if len(data) >= lim:
                        break
        return data
            
    def harris(self,gray):
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray,2,3,0.04)
        
        # Threshold for an optimal value, it may vary depending on the image.
        ret, dst = cv2.threshold(dst,self.TSH*dst.max(),255,0)
        
        dst = np.uint8(dst)
        return dst

if __name__ == "__main__":
    #print cv2.__version__
    #my current = v 2.4.8
    oi = OpenImages()
    print oi.get_class_names()
    print oi.get_all_labels(lim=4)
    print oi.get_fold_images(lim=3)
    oi.sample_images(show=True, lim=4)
    full_imlist = oi.single_folds(2, lim=3)
    imagelist = full_imlist[0]
    print full_imlist
    print imagelist
    for ima in imagelist:
        img = oi.open_image(ima)
        oi.show_image(img, ima)
        img = oi.make_gray(img)
        oi.show_image(img, ima)
        himg = oi.harris(img)
        oi.show_image(himg, ima)

    folds = oi.get_folds(lim=3)
    print folds
    print folds[0]
    for ima in folds[0]:
        img = oi.open_image(ima)
        oi.show_image(img, ima)
        img = oi.make_gray(img)
        oi.show_image(img, ima)
        himg = oi.harris(img)
        oi.show_image(himg, ima)
