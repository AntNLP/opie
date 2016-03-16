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
    fig = plt.figure()
    for  i in range(1, 5):
        pickle_content = r"../../data/domains/" + domains[i-1]+ r"/pickles/"
        y_test_have_general = np.load(pickle_content+"y_test_have_general.npy")
        y_proba_have_general = np.load(pickle_content+"y_proba_have_general.npy")
        y_test_no_general = np.load(pickle_content+"y_test_no_general.npy")
        y_proba_no_general = np.load(pickle_content+"y_proba_no_general.npy")
        fig.add_subplot(2, 2, i)
        precision, recall, thresholds = precision_recall_curve(y_test_have_general, y_proba_have_general[:, 1])
        plt.plot(recall, precision, label="P-R curve")
        precision, recall, thresholds = precision_recall_curve(y_test_no_general, y_proba_no_general[:, 1])
        plt.plot(recall, precision, label="P-R curve")
        plt.xlabel('Recall')
        plt.ylabel('Precision')
    plt.show()




