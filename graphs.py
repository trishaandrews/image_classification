import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr

from sklearn.metrics import confusion_matrix

sns.set(style="whitegrid", color_codes=True)

def make_accuracy_graphs(k, datafile, titlefront="", show=False):
    limdata = datafile[datafile["k"] == k]
    gnb = limdata[limdata["model"] == 'gnb']
    rf = limdata[limdata["model"] == 'rf']
    tree = limdata[limdata["model"] == 'tree']
    mnb = limdata[limdata["model"] == 'mnb']
    logreg = limdata[limdata["model"] == 'logreg']
    
    ax = plt.subplot(111)
    plt.plot(rf["lim"], rf["accuracy"], label='rf', color="green")
    plt.plot(tree["lim"], tree["accuracy"], label='tree', color="red")
    plt.plot(gnb["lim"], gnb["accuracy"], label='gnb', color="blue")
    plt.plot(logreg["lim"], logreg["accuracy"], label='logreg', color="orange")
    plt.plot(mnb["lim"], mnb["accuracy"], label='mnb', color="purple")
    ax.legend(loc='best')
    plt.xlabel("number training images")
    plt.ylabel("accuracy")
    plt.title("k=%s clusters" %k)
    plt.savefig(titlefront + "accuracies_k=%s.jpg"%k)
    if show:
        plt.show()

def run_accuracies(show=False):
    filename = "runallnew8000.data"
    datafile = pd.read_csv(filename, sep=" ")
    for k in [10,50,100,300]:
        make_accuracy_graphs(k, datafile, titlefront="test8000_", show=show)
    #plt.show()

def make_confustion_matrix(y_test, y_pred):
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    print cm
    
    # Show confusion matrix in a separate window
    plt.matshow(cm)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
