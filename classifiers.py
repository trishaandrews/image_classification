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
    images = []
    labels = []
    #print tuplelist[1]
    for tl in tuplelist:
        images.append(tl[0])
        labels.append(oi.get_label(tl[1]))
    return images, labels

def select_training(oi, folds=False, lim=None):
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
        #the above is showing individual image, image label
    else:
        X_train = oi.get_all_images(lim=lim)
        y_train = oi.get_all_labels(lim=lim)
        #trainoi.show_image(X_train_full[0], 0)
        #x_train[i] is an individual image array
    return X_train, y_train, folds

def verify(oi, x, y, lim, folds):
    #print x
    #print y
    if folds:
        for i, f in enumerate(x):
            for t in range(lim):
                oi.show_image(x[i][t], label=y[i][t])
    else:
        for t in range(lim):
            #print "yt:", y[t]
            oi.show_image(x[t], label=y[t])
def make_harris(xs, oi):
    xsfinal = []
    for x in xs:
        img = oi.harris(x, grey=False)
        xsfinal.append(img)
    xsfinal = np.asarray(xsfinal)
    return xsfinal

def make_flat(xs, oi):
    xsf = []
    for x in xs:
        x = x.flatten()
        #print type(x)
        xsf.append(x)
    print "x shape", x.shape
    xsfinal = np.asarray(xsf)
    return xsfinal

def flatmodels():
    X_train, y_train, folds = select_training(trainoi, lim=lim)
    #X_train = make_harris(X_train, trainoi)
    #verify(trainoi, X_train, y_train, lim, folds)
    X_train = make_flat(X_train, trainoi)
    X_test = testoi.get_all_images(lim=lim) #x_test[i] is an individual image array
    y_test = testoi.get_all_labels(lim=lim)
    #X_test = make_harris(X_test, testoi)
    #verify(testoi, X_test, y_test, lim, folds)
    X_test = make_flat(X_test, testoi)

    #print len(X_train), len(X_train[0])
    print X_train.shape
    print X_train[0].shape
    #print X_train[0][0].shape
    #print y_train[0]
    print y_train.shape

def run_models(X_train, y_train, X_test, y_test, models, label):
    for mname, m in models.iteritems():
        print "*** %s" % mname
        t0 = time.time()
        if mname == 'kmeans':
            m.fit(X_train)
        else:
            m.fit(X_train, y_train)
        #pred_probs[mname] = {'train': m.predict_proba(X_train),  
        #                     'test': m.predict_proba(X_test)}
        #pred = m.predict(X_test)
        #prec, recall, fscore, sup = precision_recall_fscore_support(y_test, pred)
        #scores[mname] = {'accuracy': accuracy_score(y_test, pred),
        #                 'precision': prec,
        #                 'recall': recall,
        #                 'fscore': fscore}
        t1 = time.time()
        title = mname + "_" + str(label) + "_" + time.strftime("%d-%b-%Y-%H:%M") +".pkl"
        pickle_stuff(title, m)
        print "total time: {:.2f}".format(t1-t0)

    for s, score in scores.iteritems():
        print s
        for n, num in score.iteritems():
            print n ,":", num
            #print scores

def run_kmeans(X_train, models, label):
    for mname, m in models.iteritems():
        print "*** %s" % mname
        t0 = time.time()
        if mname == 'kmeans':
            m.fit(X_train)
        else:
            break
        t1 = time.time()
        title = mname + "_" + str(label) + "_" + time.strftime("%d-%b-%Y-%H:%M") +".pkl"
        pickle_stuff(title, m)
        print "total time: {:.2f}".format(t1-t0)

trainoi = OpenImages()
testoi = OpenImages(xfl="test_X.bin", yfl="test_y.bin")

models = {#'logistic': LogisticRegression(dual=True),
          #'rf': RandomForestClassifier(),
          'kmeans' : KMeans(n_clusters=200, n_init=15, verbose=1, n_jobs=6),
          #'knn': KNeighborsClassifier(),
          #'svc': SVC(probability=True, verbose=True),
          #'tree': DecisionTreeClassifier(),
          #'gnb': GaussianNB()
}

pred_probs = {}
preds = {}
scores = {}


lim = 100
for i in range(1,11):
    label = i #9 is ships
    #ship_folds = trainoi.single_folds(9, lim=lim) 
    #ship_single_inds = ship_folds[0] #just the first fold for now
    
    ships = trainoi.get_all_images(classiftype=label, lim=lim)
    print len(ships)
    shipys = [label for l in range(len(ships))]
    #for s in ship_single_inds:
    #    im = trainoi.open_image(s)
    #    label = trainoi.get_label(s)
    #    ships.append(im)
    #    shipys.append(label)
    
    #print type(ship_single[0])
    
    siftdesc = trainoi.many_sifts(ships, grey=False)
    
    #X_test = testoi.get_all_images(lim=lim) #x_test[i] is an individual image array
    #y_test = testoi.get_all_labels(lim=lim)
    #X_test = make_harris(X_test, testoi)
    #verify(testoi, X_test, y_test, lim, folds)
    #X_test = make_flat(X_test, testoi)
    #X_test = testoi.many_sifts(X_test, grey=False)
    
    print siftdesc.shape
    run_kmeans(siftdesc, models, label) 
#trainoi.cv2_kmeans(siftdesc, 200, 15)

#(n_clusters=8, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1)
'''
for mname, m in models.iteritems():
    print "*** %s" % mname
    m.fit(X_train, y_train)
    pred_probs[mname] = {'train': m.predict_proba(X_train),  
                         'test': m.predict_proba(X_test)}
    pred = m.predict(X_test)
    prec, recall, fscore, sup = precision_recall_fscore_support(y_test, pred)
    scores[mname] = {'accuracy': accuracy_score(y_test, pred),
                     'precision': prec,
                     'recall': recall,
                     'fscore': fscore}

#if __name__=="__main__":
#    oi = OpenImages()
#    #oi.sample_images(show=True)
'''
