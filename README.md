# Shrinkage Discriminant Analysis
Provides an efficient framework for high-dimensional linear and diagonal discriminant analysis with variable selection. The classifier is trained using James-Stein-type shrinkage estimators and predictor variables are ranked using correlation-adjusted t-scores (CAT scores). Variable selection error can be controlled using false non-discovery rates or higher criticism.

Python port of https://cran.r-project.org/web/packages/sda/index.html with scikit-learn compatibile class definition.

Testing
~~~~~~~
    import shrinkage_da
    mymodel = shrinkage_da.ShrinkageDiscriminantAnalysis()
    n_by_p_Xtrain = ...
    ytrain = ...
    n_by_p_Xtest = ...
    mymodel.fit(n_by_p_Xtrain, ytrain)
    mymodel.predict(n_by_p_Xtest)

Citation:
Ahdesmäki, A., and K. Strimmer. 2010.  Feature selection in omics prediction problems using cat scores and false non-discovery rate control. [Ann. Appl. Stat. 4: 503-519](https://projecteuclid.org/DPubS?service=UI&version=1.0&verb=Display&handle=euclid.aoas/1273584465). Preprint available from http://arxiv.org/abs/0903.2003.

