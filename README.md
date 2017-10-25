# hierreg

This is a Python package intended for fitting linear models whose coefficients can have some deviation according to groups to which observations belong, in a similar way as random effects and hierarchical bayes models, but following more of a ‘statistical leaning’ procedure for the estimation, by applying regularization to the random effects (group deviations).

An example such application would be to estimate a model to predict some variable of interest based on predictors/covariates collected at different physical locations (e.g. students from different schools, sales form different stores, etc.), or survey answers from different people, when data from these different groups (people/schools/regions/etc.) doesn’t behave the same and it’s desired to have different coefficients for each, but with these coefficients still being close to each other.

Under the package’s default settings, the loss to minimize is as follows:

```
L(w, v) = norm( y - X*(w + sum_groups(v_group*I[x in group]) )/sqrt(nobs) + reg_param*norm(group_weights*v)
```

Where:

	* 'X' is the predictors/covariates matrix
	* 'y' is the value to predict
	* 'w' are the coefficients for each variable
	* 'v_group' are deviations from those coefficients for each group (1 variable per coefficient per group)
	* ‘group_weights' are weights for the deviation for each group, summing up to 1, and inversely proportional to the number of observations coming from each group
	* 'I[x in group]' is an indicator column-matrix with ones when a row of X belongs to a given group
	
While this doesn’t provide the same inference power or statistical testing capabilities as hierarchical bayes or random random effects models, respectively, it can sometimes result in models that are able to make better predictions than either of them (see references at the bottom).

## Installation
Package is available on PyPI, can be installed with

```pip install hierreg```

## Usage
Example usage

```
import numpy as np, pandas as pd
from hierreg import HierarchicalRegression

## Generating sample data
X=np.random.normal(size=(1000,3))
groups=pd.get_dummies(np.random.randint(low=1,high=4,size=1000).astype('str')).as_matrix()

## Making a variable dependent on the input
y=(37+2*groups[:,0]-4*groups[:,2]) + X[:,0]*(0+2*groups[:,1]-groups[:,0]) + 0.5*X[:,1] + 7*X[:,2] - 2*X[:,1]**2

## Adding noise
y+=np.random.normal(scale=5,size=1000)

## Fitting the model
hlr=HierarchicalRegression()
hlr.fit(X,y,groups)

## Making predictions
yhat=hlr.predict(X,groups)
```

A more detailed example with real data can be found under [this IPython notebook](http://nbviewer.ipython.org/github/david-cortes/hierreg/blob/master/example/hierreg_example.ipynb)

`HierarchicalRegression` can also fit a logistic regression model when called with argument `problem='classification'`.

## Notes
The number of parameters to estimate can get very large, especially when there are many categorical variables. While a non-hierarchical model would have `nfeatures` parameters, this model will have `nfeatures*(1+ngroups)` parameters, so be aware of large memory usage.

As a lot more parameters are determined with this type of model, poor tuning of the regularization parameter can result in models that perform a lot worse than simpler non-hierarchical models.

The model is fit by interfacing some solver through an optimization modeling framework (either cvxpy-->scs/ecos/cvxopt or casadi-->ipopt), so speed will not be great when dealing with large datasets.

## References
* Evgeniou, T., Pontil, M., & Toubia, O. (2007). A convex optimization approach to modeling consumer heterogeneity in conjoint estimation. Marketing Science, 26(6), 805-818.
* Ando, R. K., & Zhang, T. (2005). A framework for learning predictive structures from multiple tasks and unlabeled data. Journal of Machine Learning Research, 6(Nov), 1817-1853.
