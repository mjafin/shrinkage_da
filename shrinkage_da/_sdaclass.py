"""
Class for Shrinkage Discriminant Analysis using James-Stein shrinkage
"""
#import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

from predict_sda import predict_sda
from sda import sda
from sda_ranking import sda_ranking

class ShrinkageDiscriminantAnalysis(BaseEstimator, ClassifierMixin):
    """ Shrinkage Discriminant Analysis using James-Stein shrinkage
    
    A classifier with a linear decision boundary, generated by fitting class
    conditional densities to the data using Bayes' rule and applying shrinkage
    to the variance, correlation and class prior estimates.

    The model fits a Gaussian density to each class, assuming that all classes
    share the same covariance matrix. Any number of classes, two or more, supported.

    Due to not explicitly evaluating the correlation matrix this implementation
    is very fast for large dimensional input data provided that the data is either
    n>>p or p>>n as is typical in e.g. genomics.

    Parameters
    ----------
    lambda_var : float or list, default=None
        Shrinkage parameter for variances or list specified_lambda_var = lambda_varif separate ones used per class.
    lambda_freqs : float, default=None
        Shrinkage parameter for class prevalences. Estimated from data if None.
    diagonal : bool, default=False
        If True, skip correlation adjustment and assume diagonal model
    ranking_score : string, default = 'entropy'
        One of "entropy", "avg" or "max". For binary classification the choices
        converge to the same result, thus only important for multi class
        feature ranking scores
    verbose : bool, default=False
        Verbose mode.
    

    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.
    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.
    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.
    """
    def __init__(self, lambda_cor = None, lambda_var = None, lambda_freqs = None, diagonal=False, ranking_score = 'entropy', verbose=False):
        self.lambda_cor = lambda_cor
        self.lambda_var = lambda_var
        self.lambda_freqs = lambda_freqs
        self.diagonal = diagonal
        self.ranking_score = ranking_score
        self.verbose = verbose
        

    def fit(self, X, y):
        """Fit ShrinkageDiscriminantAnalysis model according to the given
           training data and parameters.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int or list of class labels.

        Returns
        -------
        self : object
            Returns self.
        """
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y
        # Return the classifier
        self.sdamodel_ = sda(Xtrain=X, L=y, lambda_cor = self.lambda_cor, 
                            lambda_var = self.lambda_var, lambda_freqs = self.lambda_freqs, 
                            diagonal = self.diagonal, verbose = self.verbose)
        return self

    def predict(self, X):
        """ A reference implementation of a prediction for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample is the label of the closest sample
            seen udring fit.
        """
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_', 'sdamodel_'])

        # Input validation
        X = check_array(X)
        my_preds = predict_sda(sda_object = self.sdamodel_, Xtest = X, verbose = self.verbose)
        return my_preds["predicted_class"]
    
    def predict_proba(self, X):
        """Return posterior probabilities of classification.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Array of samples/test vectors.
        Returns
        -------
        C : array, shape = [n_samples, n_classes]
            Posterior probabilities of classification per class.
        """

        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)
        
        my_preds = predict_sda(sda_object = self.sdamodel_, Xtest = X, verbose = self.verbose)
        return my_preds["posterior"]

    def feature_rank(self, X, y):
        """Rank features utilising correlation adjusted t-scores using the given
           training data and parameters.
           
           _self.rankings[""]

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int or list of class labels.

        Returns
        -------
        self : object
            Returns self.
        """
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y
        self.rankings_ = sda_ranking(Xtrain=X, L=y, lambda_cor = self.lambda_cor, 
                                   lambda_var = self.lambda_var, lambda_freqs = self.lambda_freqs, 
                                   ranking_score = self.ranking_score, diagonal = self.diagonal, verbose = self.verbose)
        # Return the classifier
        return self