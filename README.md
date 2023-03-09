# KNN and LMNN,NCA algorithms from scratch
## Overview
This project implements two algorithms, K-Nearest Neighbors (KNN) and Large Margin Nearest Neighbor (LMNN) using the Neighbourhood Component Analysis (NCA) approach. The KNN algorithm classifies new data points based on the 'k' closest neighbors in the training dataset, while LMNN-NCA is a distance-based algorithm that aims to learn a linear transformation of the features to better separate data points of different classes.

Both algorithms are implemented in Python from scratch without using any machine learning libraries except for pandas, numpy, math, and collections.

## Dataset

The `wine.csv` dataset is used in this project. It contains the results of a chemical analysis of wines grown in a particular region in Italy. The first column is the class label, which represents the origin of the wine, and the remaining columns are the features of the wine. The script reads a dataset `wine.csv` and splits it into training and testing data using an 80-20 split ratio. Then, for each test point, the algorithm predicts the label and computes the accuracy and confusion matrix.

## Usage

### KNN algorithm
The KNN algorithm is a supervised machine learning algorithm used for classification problems. It classifies an unknown point by finding the K nearest points in the training dataset and taking a majority vote of their classes. The `KNN.py` script implements the KNN algorithm from scratch without using any external libraries.
- `featuresTrain`: training set features
- `labelsTrain`: training set labels
- `featureTest`: test set features
- `k`: the number of neighbors to consider (default value is 15)

The `trainTestSplit()` function is used to split the data into training and testing sets. The `confusionMatrix()` function is used to calculate the confusion matrix. The `KNNPredict()` function is used to predict the class of the test set.

### LMNN-NCA algorithm
The LMNN and NCA algorithms are distance metric learning algorithms used for classification tasks. They learn a transformation that maps the input space into a more discriminative space, where examples from the same class are closer to each other and examples from different classes are farther apart. The `LMNN_NCA.py` script implements the LMNN and NCA algorithms using the `metric_learn` library.
- `features`: feature matrix
- `labels`: class labels

The `trainTestSplit()` function is used to split the data into training and testing sets. The `confusionMatrix()` function is used to calculate the confusion matrix. The `KNNPredict()` function is used to predict the class of the test set.

## Conclusion

These scripts demonstrate the implementation of KNN and LMNN algorithms for classification tasks. The KNN algorithm is a simple but effective algorithm that can achieve high accuracy on small to medium-sized datasets. The LMNN and NCA algorithms are more complex but can learn a better distance metric for high-dimensional datasets, resulting in improved classification performance.

## References

- Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification. IEEE Transactions on Information Theory, 13(1), 21-27.
- Weinberger, K. Q., & Saul, L. K. (2009). Distance metric learning for large margin nearest neighbor classification. Journal of Machine Learning Research, 10(2), 207-244.
- Goldberger, J., Roweis, S., Hinton, G., & Salakhutdinov, R. (2004). Neighbourhood components analysis. In Advances in neural information processing systems (pp. 513-520).
