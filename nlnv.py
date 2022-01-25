# Imports
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
import scipy.signal
import scipy.stats
from scipy.stats import lognorm, norm
from scipy.special import erfcinv

# Normal with log-normal variance distribution
class NLNV(scipy.stats.rv_continuous):
    def _pdf_prevec(self, v, lnvmu, lnvsigma):
        # print("pdf_prevec")
        return integrate.quad(lambda x : lognorm.pdf(x, lnvsigma, np.exp(lnvmu)) * norm.pdf(v, 0, np.sqrt(x)), 0, np.inf)[0]
    def _pdf(self, v, lnvmu, lnvsigma):
        print("pdf")
        return np.vectorize(self._pdf_prevec)(v, lnvmu, lnvsigma)
    def _cdf_prevec(self, v, lnvmu, lnvsigma):
        print("cdf_prevec")
        return integrate.quad(lambda x : lognorm.pdf(x, lnvsigma, np.exp(lnvmu)) * norm.cdf(v, 0, np.sqrt(x)), 0, np.inf)[0]
    def _cdf(self, v, lnvmu, lnvsigma):
        print("cdf")
        return np.vectorize(self._cdf_prevec)(v, lnvmu, lnvsigma)
    def _ppf_prevec(self, v, lnvmu, lnvsigma):
        print("ppf_prevec")
        return integrate.quad(lambda x : lognorm.pdf(x, lnvsigma, np.exp(lnvmu)) * norm.ppf(v, 0, np.sqrt(x)), 0, np.inf)[0]
    def _ppf(self, v, lnvmu, lnvsigma):
        print("ppf")
        return np.vectorize(self._ppf_prevec)(v, lnvmu, lnvsigma)
    def _argcheck(self, lnvmu, lnvsigma):
        return lnvsigma > 0
    def _fitstart(self, data):
        print("fitstart")
        return np.var(data), np.var(data)/100, np.mean(data), 1.0

# Test NLNV
if __name__ == '__main__':
    # Test with uniformly sampled data
    samps = np.random.uniform(10, 15, 40)

    # Create and fit distribution
    nlnv = NLNV()
    sparams = nlnv.fit(samps, fscale=1)
    print("NLNV fitted parameters:", sparams)

    # Probability plot of NLNV
    results = scipy.stats.probplot(samps, sparams=sparams, dist=NLNV(), plot=plt, fit=True)
    print(results)
    plt.show()

    # Probability plot of normal distribution
    results = scipy.stats.probplot(samps, dist='norm', plot=plt)
    print(results)
    plt.show()

    # Probability plot of uniform distribution
    results = scipy.stats.probplot(samps, dist='uniform', plot=plt)
    print(results)
    plt.show()
