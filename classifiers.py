import time
import pickle
import numpy as np

#from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
#from sklearn.learning_curve import learning_curve
#from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import scale
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_curve, auc

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

from open_images import OpenImages

#class Classifiers:

#    def __init__(self):
#        self.oi = OpenImages()

def pickle_stuff(filename, data):
    ''' open file '''
    with open(filename, 'w') as picklefile:
        pickle.dump(data, picklefile)

def unpickle(filename):
    ''' save file '''
    with open(filename, 'r') as picklefile:
        old_data = pickle.load(picklefile)
    return old_data

def separate_image_label(oi, tuplelist):
    ''' list of tuples of (image, offset). separate into lists of image arrays
    and class labels'''
    images = []
    labels = []
    for tl in tuplelist:
        images.append(tl[0])
        labels.append(oi.get_label(tl[1]))
    return images, labels

def select_training(oi, folds=False, lim=None):
    '''extract desired images for training data'''
    X_train = []
    y_train = []
    if folds:
        X_train_folds = oi.get_fold_images(lim=lim)
        for xtf in X_train_folds:
            images, labels = separate_image_label(oi, xtf)
            X_train.append(images)
            y_train.append(labels)
        #print len(X_train_folds) #len of list of folds
        #print len(X_train_folds[0]) #len of first fold
        #print type(X_train_folds[0][0]) #tuple of offset, image
        #trainoi.show_image(X_train_folds[0][0][0], X_train_folds[0][0][1]) 
        #the above would show individual image, image label
    else:
        X_train = oi.get_all_images(lim=lim)
        y_train = oi.get_all_labels(lim=lim)
        #x_train[i] is an individual image array
    return X_train, y_train, folds

def verify(oi, x, y, lim, folds):
    '''display a given training image (x) and its label (y). 
    necessary limit on number of images to display and folds status'''
    if folds:
        for i, f in enumerate(x):
            for t in range(lim):
                oi.show_image(x[i][t], label=y[i][t])
    else:
        for t in range(lim):
            oi.show_image(x[t], label=y[t])

def make_harris(xs, oi):
    ''' get harris corner features for the given list of images (xs)'''
    xsfinal = []
    for x in xs:
        img = oi.harris(x, grey=False)
        xsfinal.append(img)
    xsfinal = np.asarray(xsfinal)
    return xsfinal

def make_flat(xs, oi, verbose=1):
    ''' convert the (x,y,chan) numpy array image into a flat (x*y*chan) 
    numpy array'''
    xsf = []
    for x in xs:
        x = x.flatten()
        xsf.append(x)
    if verbose == 1:
        print "new x shape", x.shape
    xsfinal = np.asarray(xsf)
    return xsfinal

def flatmodels(verbose=True, harris=False, verify=False, lim=None):
    X_train, y_train, folds = select_training(trainoi, lim=lim)
    X_test = testoi.get_all_images(lim=lim) #x_test[i] is an individual image array
    y_test = testoi.get_all_labels(lim=lim)
    if harris:
        X_train = make_harris(X_train, trainoi)
        X_test = make_harris(X_test, testoi)
    if verify and lim:
        verify(trainoi, X_train, y_train, lim, folds)
        verify(testoi, X_test, y_test, lim, folds)
    elif verify and not lim:
        print "please enter a number of images to display"
    X_train = make_flat(X_train, trainoi)
    X_test = make_flat(X_test, testoi)
    if verbose:
        print "X train shape:", X_train.shape
        print "X train image shape:", X_train[0].shape
        print "y train label shape:", y_train.shape

def run_models(X_train, y_train, X_test, y_test, models, label):
    for mname, m in models.iteritems():
        print "*** %s" % mname
        t0 = time.time()
        m.fit(X_train, y_train)
        pred_probs[mname] = {'train': m.predict_proba(X_train),  
                             'test': m.predict_proba(X_test)}
        pred = m.predict(X_test)
        prec, recall, fscore, sup = precision_recall_fscore_support(y_test, pred)
        scores[mname] = {'accuracy': accuracy_score(y_test, pred),
                         'precision': prec,
                         'recall': recall,
                         'fscore': fscore}
        t1 = time.time()
        title = mname + "_" + str(label) + "_" + time.strftime("%d-%b-%Y-%H:%M") +".pkl"
        pickle_stuff(title, m)
        print "total time: {:.2f}".format(t1-t0)

    for s, score in scores.iteritems():
        print s
        for n, num in score.iteritems():
            print n ,":", num

def run_kmeans(X_train, models, label, lim, k):
    for mname, m in models.iteritems():
        print "*** %s" % mname
        t0 = time.time()
        if mname == 'kmeans':
            m.fit(X_train)
        else:
            break
        t1 = time.time()
        title = mname + "_" + str(label) + "_lim" + str(lim) + "_k" + str(k) + ".pkl"
        # "_" + time.strftime("%d-%b-%Y-%H:%M") +".pkl"
        pickle_stuff(title, m)
        print "total time: {:.2f}".format(t1-t0)

def run_many_kmeans(lims_=[10, 50, 100, 300, 500], ks_=[10, 50, 100, 300, 500]):
    ''' run a series of kmeans clustering models with different numbers of 
    training photos and clusters'''
    lims = lims_
    ks = ks_
    for lim in lims:
        for k in ks:
            for i in range(2,12):
                if i <=10:
                    label = i #9 is ships
                else:
                    label = None
                models = {'kmeans' : KMeans(n_clusters=k, n_init=15, n_jobs=6)}
                ships = trainoi.get_all_images(classiftype=label, lim=lim)
                print "num photos: ", len(ships)
                print "k: ", k
                print "class", str(label)
                shipys = [label for l in range(len(ships))]
                siftdesc = trainoi.many_sifts(ships, grey=False)
                print siftdesc.shape
                run_kmeans(siftdesc, models, label, lim, k) 

def retrieve_label_kmns(lim=50, k=100):
    '''retrieve trained kmeans models for each image class using given numbers 
    of training images (lim) and clusters (k)'''
    #lim = lim_
    #k = k_
    kmnmodels = {}
    kmncenters = {}
    for c in range(1,11):
        label = str(c)
        name = "kmeans" + label
        filen = kmeansdir + label + "/kmeans_" + label + "_lim" + str(lim) + "_k" + str(k) + ".pkl"
        kmnmodels[name] = unpickle(filen)
        kmncenters[name] = kmnmodels[name].cluster_centers_
    return kmnmodels, kmncenters

kmeansdir = "./kmeans_cl/kmeans_cl"

trainoi = OpenImages()
testoi = OpenImages(xfl="test_X.bin", yfl="test_y.bin")

#models = {#'logistic': LogisticRegression(dual=True),
#          #'rf': RandomForestClassifier(),
#          #'knn': KNeighborsClassifier(),
#          #'svc': SVC(probability=True, verbose=True),
#          #'tree': DecisionTreeClassifier(),
#          #'gnb': GaussianNB()
#}

pred_probs = {}
preds = {}
scores = {}

kmnmodels, centers = retrieve_kmns()
print centers["kmeans2"]

off = 0
testimage = testoi.open_image(off)
testlabel = testoi.get_label(off)
                



'''
#if __name__=="__main__":
#    oi = OpenImages()
#    #oi.sample_images(show=True)
'''
