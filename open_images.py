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

    def show_image(self, img, off=None, label=None):
        if off:
            cv2.imshow(str(self.get_label(off)), img)
        elif label:
            cv2.imshow(str(label), img)
        else:
            cv2.imshow("image", img)
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
            
    def get_all_images(self, show=False, classiftype=None, lim=0):
        images = []
        with open(self.xfile, 'rb') as f:
            count = 0
            for chunk in iter(partial(f.read, self.IMCHSIZE), ''):
                #print "clftype", classiftype
                #print self.get_label(count)
                if (classiftype and (self.get_label(count) == classiftype)) or not classiftype:
                    img = np.zeros((self.IMDIM, self.IMDIM, self.CHAN), 
                                   np.uint8)
                    for e in range(self.CHAN):
                        a = np.fromstring(chunk[e*self.IMSIZE:(e+1)*
                                                self.IMSIZE], dtype=np.uint8)
                        im = a.reshape((self.IMDIM, self.IMDIM)).transpose()
                        img[:,:,-(e+1)] = im
                    images.append(img)
                if show:
                    self.show_image(img, count)
                count +=1
                if lim != 0:
                    if len(images) >= lim:
                        break
        images = np.asarray(images)
        return images

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

    def get_folds(self, lim=0):
        fis = []
        with open(self.ffile, 'r') as inf:
            for line in inf:
                intlist = [int(l) for l in line.split()]
                if lim != 0:
                    fis.append(list(np.random.choice(intlist, lim)))
                else:
                    fis.append(intlist)
        folds = np.asarray(fis)
        return folds

    def single_folds(self, classiftype, lim=0):
        fis = []
        with open(self.ffile, 'r') as inf:
            for line in inf:
                flist = []
                for l in line.split():
                    ind = int(l)
                    if self.get_label(ind) == classiftype:
                        flist.append(ind)
                    if lim != 0:
                        if len(flist) >= lim:
                            break
                fis.append(flist)
        folds = np.asarray(fis)
        return folds

    #def get_single_class_images(self, classiftype, lim=None):
        
        
    def get_all_image_folds(self, lim=0):
        fold_data = []
        folds = self.get_folds(lim)  #([f0],[f1],...)
        for i, f in enumerate(folds): #[f0],[f1],..
            fold_num = i
            for off in f: #f1 = [0, 400, 3845, 2, ...]
                fold_data.append((off, fold_num))
        return fold_data

    def get_fold_images(self, lim=0):
        fold_data = []
        folds = self.get_folds(lim)
        for f in folds:
            fold = []
            for off in f:
                fold.append((self.open_image(off), off))
            fold_data.append(fold)
            fold_data = np.asarray(fold_data)
        return fold_data
    
    def get_all_labels(self, lim=0):
        labels = []
        with open(self.yfile, 'rb') as df:
            for i in iter(partial(df.read, 1), ''):
                name = np.fromstring(i, dtype=np.uint8)
                nm =  name[0].astype(int)
                labels.append(nm)
                if lim != 0:
                    if len(labels) >= lim:
                        #print labels
                        break
        labels = np.asarray(labels)
        return labels

    '''
    def get_all_labels(self, lim=None):
        with open(self.xfile, 'rb') as df:
            print self.xfile
            print self.yfile
            data = []
            index = 0
            for i in iter(partial(df.read, 1), ''):
                label = self.get_label(index)
                data.append(label)
                index += 1
                if lim:
                    if len(data) >= lim:
                        break
        return data
     '''
    
    def harris(self, gray, grey=True):
        if not grey:
            gray = self.make_gray(gray)
            
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray,2,3,0.04)
        
        # Threshold for an optimal value, it may vary depending on the image.
        ret, dst = cv2.threshold(dst,self.TSH*dst.max(),255,0)
        
        dst = np.uint8(dst)
        return dst

    def flatten(self, img):
        img.flatten()
        return img
        
    def sift(self, gray):
        # detect Difference of Gaussian keypoints in the image
        detector = cv2.FeatureDetector_create("SIFT")
        #print "created sift detector"
        kps = detector.detect(gray)
        #print "detected"
        # extract normal SIFT descriptors
        extractor = cv2.DescriptorExtractor_create("SIFT")
        (kps, descs) = extractor.compute(gray, kps)
        #print "SIFT: kps=%d, descriptors=%s " % (len(kps), descs.shape)
        img = cv2.drawKeypoints(gray, kps) 
        
        #return descs

        # extract RootSIFT descriptors
        #rs = RootSIFT()
        #(kps, descs) = rs.compute(image, kps)
        #print "RootSIFT: kps=%d, descriptors=%s " % (len(kps), descs.shape)
        return img, descs

    def many_sifts(self, trainimgs, grey=True):
        descriptors = np.array([])
        for t in trainimgs:
            if not grey:
                #print t
                #print type(t)
                #self.show_image(t)
                t = self.make_gray(t)
            img, descs = self.sift(t)
            descriptors = np.append(descriptors, descs)
        desc = np.reshape(descriptors, (len(descriptors)/128, 128))
        desc = np.float32(desc)
        return desc
            
    def cv2_kmeans(self, desc, k, att):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, att, 
                    1.0)
        compactness, best_labels, centers = cv2.kmeans(desc, k, criteria, 
                                                       att, flags=cv2.KMEANS_RANDOM_CENTERS)
        print "compactness", compactness
        print "best labels", best_labels
        print "centers", centers

if __name__ == "__main__":
    #print cv2.__version__
    #my current = v 2.4.8
    oi = OpenImages()
    #print oi.get_class_names()
    #print oi.get_all_labels(lim=3)
    #print oi.get_all_image_folds(lim=3)
    #print oi.get_fold_images(lim=3) 
    #oi.get_all_images(show=True, lim=5)
    full_imlist = oi.single_folds(9, lim=3) #9 is ships
    imagelist = full_imlist[0]
    #print full_imlist
    #print imagelist
    #desc = oi.many_sifts(full_imlist, grey=False)
    for ima in imagelist:
        img = oi.open_image(ima)
        oi.show_image(img, ima)
        gimg = oi.make_gray(img)
        oi.show_image(gimg, ima)
        himg = oi.harris(gimg)
        oi.show_image(himg, ima)
        simg, desc = oi.sift(gimg)
        oi.show_image(simg, ima)

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
        simg, desc = oi.sift(img)
        oi.show_image(simg, ima)
