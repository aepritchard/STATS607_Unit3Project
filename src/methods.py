import numpy as np
import statsmodels.api as sm
from scipy.stats import norm
from scipy.special import expit

from abc import ABC, abstractmethod

class Methods(ABC):
    '''
    Estimators for one randomized experiment
    Fits the logit model, then computes the plug-in and ITT estimators
            of the differential log odds
    '''

    def __init__(self, trt, Y_observed, covariate_observed):
        self.trt = trt
        self.Y_observed = Y_observed
        self.covariate_observed = covariate_observed

    @staticmethod
    def check_probability(prob):
        '''
        For probabilities not in (0,1), replace by NaN
        '''
        prob = np.asarray(prob, dtype=float)
        prob = np.where((prob <= 0) | (prob >= 1), np.nan, prob)

        # For probability checks for single numbers, return the number
        if prob.ndim == 0:
            return float(prob)
        return prob
    
    @staticmethod
    @abstractmethod
    def link(prob):
        '''
        Link function: logit for the logit model, Phi^{-1} for the probit model
        '''
        pass

    @staticmethod
    @abstractmethod
    def inverse_link(linear_predictor):
        '''
        Inverse link: maps the linear predictor to a probability
        '''
        pass

    def p(self, beta, trt, covariate_observed):
        '''
        Conditional probability that Y_i=1 given beta, X_i, Z_i
        '''
        linear_predictor = beta[:,0:1] + beta[:,1:2] * trt + beta[:,2:3] * covariate_observed
        return self.inverse_link(linear_predictor)

    @abstractmethod
    def fit_model(self):
        '''
        Fits model of Y on (1, trt, covariate_observed) with maximum likelihood
        Returns beta hat estimates
        '''
        pass
        

    def plug_in_estimator(self, beta_hat):
        '''
        Returns plug-in estimator for the differential log odds
        Gets average predicted probabilities (eq 10a) to get estimated differential log odds (eq 10b)
        '''
        # Get average predicted probabilities (eq 10a)
        alpha_tilde_treated = np.mean(self.p(beta_hat, 1, self.covariate_observed), axis=1)
        alpha_tilde_control = np.mean(self.p(beta_hat, 0, self.covariate_observed), axis=1)

        # Get estimated differential log odds (eq 10b)
        delta_tilde = self.link(alpha_tilde_treated) - self.link(alpha_tilde_control)
        return delta_tilde

    def itt_estimate(self):
        '''
        Returns ITT estimator of the differential log odds
        Gets ITT estimators (eq 11a) to get ITT estimated differential log odds (eq 11b)
        '''
        alpha_treat_hat = (self.Y_observed * self.trt).sum(axis=1) / self.trt.sum(axis=1)
        alpha_control_hat = (self.Y_observed * (1 - self.trt)).sum(axis=1) / (1 - self.trt).sum(axis=1)

        delta_hat = self.link(alpha_treat_hat) - self.link(alpha_control_hat)
        return delta_hat

    def fit_model(self, tol=1e-8, maxiter=40):
        '''
        Fits the model for all repetitions using Newton-Raphson
        '''
        n_sim, n_sample = self.trt.shape

        # Build 3D design matrix
        X = np.empty((n_sim, n_sample, 3))
        X[:,:,0] = 1.0 
        X[:,:,1] = self.trt
        X[:,:,2] = self.covariate_observed
        Y = self.Y_observed

        # Define starting values
        beta = np.zeros((n_sim, 3))
        converged = np.zeros(n_sim, dtype=bool)
        failed = np.zeros(n_sim, dtype=bool)
        identity = np.eye(3)

        for iteration in range(maxiter):
            # Linear predictor for each repetition
            eta = np.einsum("rni,ri->rn", X, beta)

            # Get residuals and weights from each specific model
            residual, weight = self.residual_and_weight(eta, Y)

            # Get gradient and hessian for each repetition
            gradient = np.einsum("rni,rn->ri", X, residual)
            hessian = np.einsum("rni,rn,rnj->rij", X, weight, X)

            # Find repetitions with a singular hessian (these reps will fail to converge)
            singular = np.linalg.cond(hessian) > 1e12
            failed = failed | (singular & ~converged)

            # Update all repeititions that are not singular and have not failed
            still_running = ~converged & ~failed
            hessian[~still_running] = identity
            step = np.linalg.solve(hessian, gradient[:,:,None])[:,:,0]
            step[~still_running] = 0.0
            beta = beta + step

            # Find repetitions that converged
            converged = converged | (still_running & (np.max(np.abs(step), axis=1) < tol))

            if not np.any(~converged & ~failed):
                break

        # For any repeititons that have not converged, give NaN coefficients
        beta[~converged] = np.nan

        return beta, converged

    def run_methods(self):
        '''
        Runs all methods on the stored experiment
        Returns a dict with the logit coefficients, plug-in estimate, and ITT estimate
        '''
        beta_hat, converged = self.fit_model()
        delta_tilde = self.plug_in_estimator(beta_hat)
        delta_hat = self.itt_estimate()

        results = {
            "beta1_hat": beta_hat[:,0],
            "beta2_hat": beta_hat[:,1],
            "beta3_hat": beta_hat[:,2],
            "plug_in": delta_tilde,
            "itt": delta_hat,
            "converged": converged
        }
        return results

class Methods_logit(Methods):

    @staticmethod
    def link(prob):
        prob = Methods.check_probability(prob)
        return np.log(prob / (1 - prob))

    @staticmethod
    def inverse_link(linear_predictor):
        return expit(linear_predictor)
    
    @staticmethod
    def residual_and_weight(eta, Y):
        p = expit(eta)
        residual = Y - p
        weight = p * (1-p)
        return residual, weight

class Methods_probit(Methods):

    @staticmethod
    def link(prob):
        prob = Methods.check_probability(prob)
        return norm.ppf(prob)

    @staticmethod
    def inverse_link(linear_predictor):
        return norm.cdf(linear_predictor)

    @staticmethod
    def residual_and_weight(eta, Y):
        p = norm.cdf(eta)
        p = np.clip(p, 1e-12, 1 - 1e-12)  # Make sure no 0 values to prevent division problems
        density = norm.pdf(eta)
        residual = (Y-p) * density / (p * (1-p))
        weight = density ** 2 / (p * (1-p))
        return residual, weight