#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 16/03/16 12:42:03

@author: Changzhi Sun
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve

if __name__ == "__main__":
    domains = ["reviews_Grocery_and_Gourmet_Food",
               "reviews_Movies_and_TV",
               "reviews_Cell_Phones_and_Accessories",
               "reviews_Pet_Supplies"]
    names = ["Food", "Movie", "Phone", "Pet"]
    for  i in range(1, 5):
        pickle_content = r"../../data/domains/" + domains[i-1]+ r"/pickles/"
        y_test_have_general = np.load(pickle_content+"y_test_have_general.npy")
        y_proba_have_general = np.load(pickle_content+"y_proba_have_general.npy")
        y_test_no_general = np.load(pickle_content+"y_test_no_general.npy")
        y_proba_no_general = np.load(pickle_content+"y_proba_no_general.npy")
        precision, recall, thresholds = precision_recall_curve(y_test_have_general, y_proba_have_general[:, 1])
        plt.plot(recall, precision, label="all")
        precision, recall, thresholds = precision_recall_curve(y_test_no_general, y_proba_no_general[:, 1])
        plt.plot(recall, precision, label="without sentiment word")
        plt.xlabel('Recall', fontsize=20)
        plt.ylabel('Precision', fontsize=18)
        plt.title(names[i-1], fontsize=25)
        plt.legend(loc="low left", numpoints=3)
        plt.savefig(names[i-1]+".pdf")
        plt.show()

