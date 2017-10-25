import pandas as pd, numpy as np, cvxpy as cvx
from sklearn.base import BaseEstimator
from scipy.sparse import csc_matrix, csr_matrix, hstack
from casadi import MX, nlpsol, dot, mtimes, log, sum1

class HierarchicalRegression(BaseEstimator):
    def __init__(self, l2_reg=100.0, l1_reg=0.0, linf_reg=0.0, main_l2_reg=0.0, problem='regression',
                 fit_intercept=True, weight_by_nobs=True, reweight_deviations=False, standardize=False,
                 solver_interface='cvxpy', cvxpy_opts={'solver':'SCS', 'max_iters':3000, 'verbose':False},
                 ipopt_options={"print_level":0,'hessian_approximation':'limited-memory'}):
        """
        Linear regression with group-varying coefficients
        
        Fits a linear regression with coefficients varying for different specified groups in the data,
        in a similar fashion as random effects models, but with a different problem formulation,
        following more of a 'statistical learning' approach.
        By default, the objective to minimize is as follows:
        
        L(w, v) = norm( (w + sum_groups(v_group*I[x in group]))*X - y )/sqrt(nobs) + reg_param*norm(weight_group*v_group)
        Where:
            -'X' is the predictors matrix
            -'y' is the value to predict
            -'w' are the coefficients for each variable
            -'v_group' are deviations from those coefficients for each group (1 variable per coefficient per group)
            -'weight_group' are weights for the deviation for each group, summing up to 1,
              and inversely proportional to the number of observations coming from each group
             
        When predicting new values, if any observation comes from a new group that was not in the training data,
        it will use only the 'w' coefficients.
             
        Note
        ----
        In comparison to other types of models, it seems that this type of model requires
        high regularization parameters in order to produce good cross-validated results.
        A poorly-tuned hierarchical model like this will produce way worse results than a plain linear model,
        so be careful to always cross-validate the resulting model.
        
        I would recommend using only l2 regularization as it seems to bring overall better results,
        and seems to also be able to shrink group coefficients to exactly zero. There's also the option to
        apply regularization not by taking the norm of the raw deviations, but by rescaling them through the
        inverse of a matrix with trace=1 (i.e. sum_group (v_group)'D^(-1)v_group, with D becoming another set of parameters),
        as used in some papers, but this will make the optimization objective a lot slower.
        
        The API contains an option to add regularization for the non-group coefficients too, but note
        that if such regularization is set, it needs to be a lot smaller than the regularization assigned to
        the group-varying parameters, otherwise you will end up with larger group parameters than shared parameters
        (which will most likely translate into way worse predictions).
        
        All the optimization routine is done with cvxpy, by default using SCS as the workhorse,
        so don't expect good speed when dealing with large datasets, and you might want to increase
        the maximum number of iterations when dealing with larger datasets. In order to change to
        a different solver within the cvxpy realm, you can pass something like cvxpy_opts={'solver':'CVXOPT'}.
        
        When using ipopt through casadi as the optimizaton workhorse, the norms in the loss function
        are squared and the number of observations not rooted, so you will need larger regularization parameters,
        and only the l2-norm regularizations will be considered.
        
        
        Parameters
        ----------
        l2_reg : float
            Regularization parameter for the l2-norm of the group deviations ('v' in the formula).
        l1_reg : float
            Regularization parameter for the l1-norm of the group deviations ('v') (Not recommended).
            (Ignored when passing 'solver_interface="casadi"' or 'reweight_deviations=True')
        linf_reg : float
            Regularization parameter for the lp-infite norm (a.k.a. Chebyshev norm) of the group deviations ('v').
            (The lp-infinite norm corresponds the largest absolute value in a vector)
            (Ignored when passing 'solver_interface="casadi"' or 'reweight_deviations=True')
        main_l2_reg : float
            Regularization parameter for the l2-norm of the coefficients ('w' in the formula) (Not recommended).
        problem : str
            Problem type - either 'regression' (minimizing squared loss) or 'classification' (minimizing logistic loss).
            Only binary classification is supported.
        fit_intercept : bool
            Whether to add a model intercept.
            Forced to 'True' when passing 'standardize=True'.
        weight_by_nobs : bool
            Whether to weigh the norm of the group deviatons according to the number of observations from each group
            (inversely proportional to the number of observations, summing up to 1 in total).
        reweight_deviations : bool
            Whether to reweight deviations in the regularization by the inverse of a matrix with trace=1
            (i.e. (v_group)'D^(-1)(v_group) with tr(D)=1 for all groups), instead of taking their norm.
            They will be weighted by the number specified in the parameter 'l2_reg'.
            Note that if 'weight_by_nobs' is set to True, it will apply in addition to this.
            Parameters 'l1_reg' and 'linf_reg' are ignored when setting 'reweight_deviations=True'.
            Forced to 'False' when passing 'solver_interface='casadi''.
        standardize : bool
            Whether to standardize the predictor variables (subtract the mean and divide by the standard deviation),
            so as to put them all in a similar scale (Not recommended).
        solver_interface : str
            Optimization library to use to model the problem. Can be either 'cvxpy' or 'casadi'.
            For larger datasets, it's recommended to use casadi+ipopt as it will require less memory and might run faster.
            For cvxpy, the underlying optimization library used to optimize the parameters is by default 'SCS',
            and for casadi it's 'ipopt' (using BFGS).
        cvxpy_opts : dict
            Additional options to pass to solver through cvxpy (ignored when passing 'solver_interface="casadi"').
            For details see:
            http://www.cvxpy.org/en/latest/tutorial/advanced/#solve-method-options
        ipopt_options : dict
            Options to pass to IPOPT (ignored when passing 'solver_interface="cvxpy"').
            For a list of options see https://www.coin-or.org/Ipopt/documentation/node40.html
        
        Attributes
        ----------
        coef_ : numpy array
            Model's coefficients.
            First entry corresponds to the intercept.
        coef2_ : numpy array
            Model's group deviations for each coefficient, as a flat 1-d array.
        group_effects_ : pandas data frame
            Model's group deviations for each coefficient, as a formatted pandas data frame.
        class1, class2
            Positive and negative class labels used when outputting predictions (Classification only).
        """
        self.l1_reg=l1_reg
        self.l2_reg=l2_reg
        self.linf_reg=linf_reg
        self.main_l2_reg=main_l2_reg
        assert (problem=='regression') or (problem=='classification')
        self.problem=problem
        self.fit_intercept=fit_intercept
        self.weight_by_nobs=weight_by_nobs
        self.reweight_deviations=reweight_deviations
        self.standardize=standardize
        assert solver_interface in {'cvxpy', 'casadi'}
        self.solver_interface=solver_interface
        self.cvxpy_opts=cvxpy_opts
        self.ipopt_options=ipopt_options
    
    def fit(self,X,y,groups):
        """
        Fit a hierarchical linear regression to data
        
        Parameters
        ----------
        X : pandas data frame or numpy array
            Table with predictors/covariates. Categorical variables must be properly encoded beforehand.
        y : pandas data frame or numpy array
            Values to predict.
        groups : pandas data frame or numpy array
            Groups to which each observation belongs.
            Can have more than one column if there are multiple group categories (e.g. 'province' and 'city').
            Values can be strings or numbers, will be one-hot-encoded internally.
            Observations not belonging to any group must have NA values.
        """
        ## checking input
        if type(X)==pd.core.frame.DataFrame:
            x=X.as_matrix()
            xnames=X.columns.values
        elif (type(X)==np.ndarray) or (type(X)==np.matrixlib.defmatrix.matrix):
            x=X.copy()
            xnames=None
        else:
            raise ValueError("'X' must be a pandas data frame or numpy array")
            
        if (type(y)==pd.core.frame.DataFrame) or (type(y)==pd.core.series.Series):
            yval=y.as_matrix()
        elif (type(y)==np.ndarray) or (type(y)==np.matrixlib.defmatrix.matrix):
            yval=y.copy()
        else:
            raise ValueError("'y' must be a pandas data frame, pandas series, or numpy array")
            
        if self.problem=='classification':
            classes=set(yval)
            if len(classes)!=2:
                raise ValueError("Only binary classification is supported. 'y' contains "+str(len(classes))+" values")
            if (1 in classes) and ((-1) in classes):
                self.class1=1
                self.class2=-1
            elif (1 in classes) and (0 in classes):
                yval[yval==0]=-1.0
                self.class1=1
                self.class2=0
            else:
                self.class1,self.class2=tuple(classes)
                yval[yval==class1]=1.0
                yval[yval==class2]=-1.0
            
        if type(groups)==pd.core.series.Series:
            gbin=pd.get_dummies(np.array(groups.astype('str')))
            self._gdim=1
            self._gnames=''
        elif type(groups)==pd.core.frame.DataFrame:
            self._gdim=groups.shape[1]
            self._gnames=list(groups.columns.values)
            gbin=pd.get_dummies(groups.astype('str'))
        elif (type(groups)==np.ndarray) or (type(groups)==np.matrixlib.defmatrix.matrix):
            try:
                self._gdim=groups.shape[1]
            except:
                self._gdim=1
            gr=pd.DataFrame(groups,columns=['X'+str(i) for i in range(self._gdim)]).astype('str')
            self._gnames=gr.columns.values
            gbin=pd.get_dummies(gr.astype('str'))
            del gr
        else:
            raise ValueError("'groups' must be a pandas data frame, pandas series, or numpy array")
            
        ## processing X as specified
        if self.standardize:
            xmeans=x.mean(axis=0)
            xsd=x.std(axis=0)
            x=(x-xmeans)/xsd
        
        if self.fit_intercept or self.standardize:
            x=np.hstack([np.ones((x.shape[0],1)), x])
            if xnames is not None:
                xnames=['Intercept']+list(xnames)
        nobs=x.shape[0]
        nvar=x.shape[1]
        assert gbin.shape[0]==nobs
        assert yval.shape[0]==nobs
        
        ## putting one-hot-encd' groups into a wide numpy array
        self._gbin_names=list(gbin.columns.values)
        ngroupvar=len(self._gbin_names)
        nweight=1/gbin.sum(axis=0)
        ntot=np.sum(nweight)
        nweight=nweight/ntot
        nweight=np.hstack([nweight for g in range(nvar)])/nvar
        
        gbin=csc_matrix(gbin.as_matrix())
        gbin=hstack([gbin.multiply(x[:,g].reshape(-1,1)) for g in range(nvar)])
        gbin=csr_matrix(gbin)
        
        ## modeling the problem
        if self.solver_interface=='cvxpy':
            w=cvx.Variable(nvar)
            v=cvx.Variable(ngroupvar*nvar)
            if self.problem=='regression':
                obj=cvx.norm(yval-x*w-gbin*v)/np.sqrt(nobs)
            else:
                obj=cvx.sum_entries(cvx.logistic(cvx.mul_elemwise(-yval,x*w+gbin*v)))/nobs
            if self.reweight_deviations:
                if self.main_l2_reg>0:
                    obj+=self.main_l2_reg*cvx.norm(w)
                D=cvx.Variable(nvar, nvar)
                for g in range(ngroupvar):
                    if self.weight_by_nobs:
                        obj+=l2_reg*cvx.matrix_frac(cvx.mul_elemwise(nweight[ngroupvar],v[[i for i in range(g,gbin.shape[1],ngroupvar)]]),D)
                    else:
                        obj+=l2_reg*cvx.matrix_frac(v[[i for i in range(g,gbin.shape[1],ngroupvar)]],D)
                prob=cvx.Problem(cvx.Minimize(obj), [cvx.trace(D)==1])
            else:
                if self.main_l2_reg>0:
                    obj+=self.main_l2_reg*cvx.norm(w)
                if self.l1_reg>0:
                    if self.weight_by_nobs:
                        obj+=self.l1_reg*cvx.norm(cvx.mul_elemwise(nweight,v),1)
                    else:
                        obj+=self.l1_reg*cvx.norm(v,1)
                if self.l2_reg>0:
                    if self.weight_by_nobs:
                        obj+=self.l2_reg*cvx.norm(cvx.mul_elemwise(nweight,v),2)
                    else:
                        obj+=self.l2_reg*cvx.norm(v,2)
                if self.linf_reg>0:
                    obj+=self.linf_reg*cvx.norm(v,'inf')
                prob=cvx.Problem(cvx.Minimize(obj))
            prob.solve(**self.cvxpy_opts)

            ## saving results
            self.coef_=np.array(w.value)
            self.coef2_=np.array(v.value)
            
        if self.solver_interface=='casadi':
            xvars=MX.sym('Vars',nvar*(1+ngroupvar))
            w=xvars[:nvar]
            v=xvars[nvar:]
            pred=mtimes(x,w)+mtimes(gbin,v)
            if self.problem=='regression':
                err=yval-pred
                obj=dot(err,err)/nobs
            else:
                obj=sum1(log(1+np.exp(-yval*pred)))/nobs
                
            if self.weight_by_nobs:
                regw=nweight*v
            else:
                regw=v
            obj+=self.l2_reg*dot(regw,regw)
            if self.main_l2_reg>0:
                obj+=self.main_l2_reg*dot(w,w)

            solver = nlpsol("solver", "ipopt", {'x':xvars,'f':obj},{'print_time':False,'ipopt':self.ipopt_options})
            x0=np.zeros(shape=nvar*(1+ngroupvar))
            res=solver(x0=x0)
            
            ## saving results
            self.coef_=np.array(res['x'][:nvar])
            self.coef2_=np.array(res['x'][nvar:])
        
        
        if self.standardize:
            div=np.array([1]+list(xsd))
            self.coef_=self.coef_.reshape(-1)/div
            self.coef2_=self.coef2_.reshape(-1)/np.hstack([[div[i]]*ngroupvar for i in range(nvar)])
            self.coef_[0]-=np.sum(self.coef_[1:]*xmeans)
            xmeans=[0]+list(xmeans)
            self.coef_[0]-=np.sum(self.coef2_*np.hstack([[xmeans[i]]*ngroupvar for i in range(nvar)]))
        
        self.group_effects_=pd.DataFrame(self.coef2_.reshape(nvar,ngroupvar),columns=self._gbin_names)
        if xnames is not None:
            self.group_effects_.index=xnames
    
    def predict(self,X,groups,prob=False):
        """
        Predict values using this model
        
        Parameters
        ----------
        X : pandas data frame or numpy array
            Table with predictors/covariates. Categorical variables must be properly encoded beforehand.
        groups : pandas data frame or numpy array
            Groups to which each observation belongs.
            Can have more than one column.
            Values can be strings or numbers, will be one-hot-encoded internally.
            Observations not belonging to any group must have NA values.
            Any groups that were not present in the training data will be ignored (no group-deviation applied)
        prob : bool
            Whether to output positive-class probability or just predicted class (classification only)
            
        Returns
        -------
        numpy array
            Predicted values.
        """
        ## checking input
        if type(X)==pd.core.frame.DataFrame:
            x=X.as_matrix()
        elif (type(X)==np.ndarray) or (type(X)==np.matrixlib.defmatrix.matrix):
            x=X.copy()
        else:
            raise ValueError("'X' must be a pandas data frame or numpy array")
            
        ## adding intercept if necessary
        if self.fit_intercept or self.standardize:
            x=np.hstack([np.ones((x.shape[0],1)), x])
        
        ## processing groups variable
        if type(groups)==pd.core.series.Series:
            if self._gdim!=1:
                raise ValueError("Model was fit with 1-dimensional groups. Must pass a 1-dimensional group array for predictions.")
            gbin=pd.get_dummies(np.array(groups.astype('str')))
        elif type(groups)==pd.core.frame.DataFrame:
            if list(groups.columns.values)!=self._gnames:
                if groups.shape[1]<self._gdim:
                    raise ValueError("'groups' must contain information about "+str(self._gnames))
                elif groups.shape[1]==self._gdim:
                    gr=groups.copy()
                    gr.columns=self._gnames
                else:
                    for cl in self._gnames:
                        if cl not in groups.columns.values:
                            raise ValueError("'groups' must contain information about "+str(self._gnames))
                    gr=groups[self._gnames]
                gbin=pd.get_dummies(gr.astype('str'))
            else:
                gbin=pd.get_dummies(groups.astype('str'))
        elif (type(groups)==np.ndarray) or (type(groups)==np.matrixlib.defmatrix.matrix):
            if len(groups.shape)==1:
                if self._gdim!=1:
                    raise ValueError("Model was fit with 1-dimensional groups. Must pass a 1-dimensional group array for predictions.")
                gbin=pd.get_dummies(groups.astype('str'))
            else:
                if self._gdim==1:
                    raise ValueError("Model was fit with 1-dimensional groups. Must pass a 1-dimensional group array for predictions.")
                if groups.shape[1]!=self._gdim:
                    raise ValueError("'groups' must contain information about "+str(self._gnames))
                gr=pd.DataFrame(groups,columns=self._gnames).astype('str')
                gbin=pd.get_dummies(gr.astype('str'))
                del gr
        else:
            raise ValueError("'groups' must be a pandas data frame, pandas series, or numpy array")
            
        missing_cols = set(self._gbin_names) - set(gbin.columns)
        for c in missing_cols:
            gbin[c] = 0
        gbin = gbin[self._gbin_names]
        gbin=csc_matrix(gbin.as_matrix())
        gbin=hstack([gbin.multiply(x[:,g].reshape(-1,1)) for g in range(self.coef_.shape[0])])
        
        ## applying coefficients
        pred=(x.dot(self.coef_)+gbin.dot(self.coef2_)).reshape(-1)
        
        ## returning output in the appropriate format
        if self.problem=='regression':
            return pred
        else:
            if prob:
                return 1/(1+np.exp(-pred))
            else:
                outclass=pred>=0
                pred[outclass]=self.class1
                pred[~outclass]=self.class2
                return pred
