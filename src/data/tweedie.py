import statsmodels.api as sm
import tweedie
import scipy as sp

def get_tweedie_power(ar):
    #Solve GLM with Tweedie distribution to get an estimation of phi
    res = sm.GLM(ar, ar, family=sm.families.Tweedie(link=sm.families.links.log(), var_power=1.1)).fit()
    #Optimize on p
    def loglike_p(p):
        return -tweedie.tweedie(mu=res.mu, p=p, phi=res.scale).logpdf(res._endog).sum()
    opt = sp.optimize.minimize_scalar(loglike_p, bounds=(1.05, 1.95), method='bounded',options={'maxiter':50})
    return opt.x