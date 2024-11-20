import numpy as np
import scipy

# Least Mean Square Regression
class LinearRegression:
    def __init__(self, a, b, d, n, T, w):
        self.a = a # sequence of data a_t
        self.b = b # sequence of data b_t
        self.d = d # dimension
        self.n = n # batch size
        self.D = np.sqrt(d) # D in Assumption 3.2
        
        # calcutate G in Assumption 3.3
        A_norm = 0.0
        for t in range(T):
            A_norm_tmp = np.linalg.norm(np.dot(a[t].T, a[t]), 2)
            if A_norm_tmp > A_norm:
                A_norm = A_norm_tmp
        v = w
        for i in range(d):
            if v[i]<1.0/2.0:
                v[i] = 1.0 - v[i]
        self.G = 2.0 * A_norm * np.linalg.norm(v, 2)

        # calculate lambda
        lam = scipy.sparse.linalg.eigs(self.hess(0), 1, which="SM")[0]
        self.lam_data = []
        for t in range(T):
            lam_tmp = scipy.sparse.linalg.eigs(self.hess(t), 1, which="SM")[0]
            self.lam_data.append(lam_tmp)
            if lam_tmp<lam:
                lam = lam_tmp
        self.lam = lam

        # calculate alpha
        self.alpha = self.lam / self.G**2

    def value(self, x, t): # function value
        y = 0
        for i in range(self.n):
            y += (np.dot(self.a[t][i], x) - self.b[t][i])**2
        return y / self.n
    
    def grad(self, x, t): # gradient
        y = np.zeros(self.d)
        for i in range(self.n):
            y = y + 2.0 * (np.dot(self.a[t][i], x) - self.b[t][i]) * self.a[t][i]
        return y / self.n
    
    def hess(self, t): # Hessian
        y = 2.0 * np.dot(self.a[t].T, self.a[t])
        return y



# Contaminated exp-concave function
class ContaminatedExpConcave:
    def __init__(self, T, k, seed):
        self.T = T # number of rounds
        self.k = k # number of contaminated rounds
        self.D = 1.0 # D in Assumption 3.2
        self.G = 100.0 # G in Assumption 3.3

        # generate contaminated rounds
        self.lam_data = np.ones(T) * self.G**2
        self.lam = self.G**2
        np.random.seed(seed)
        self.contami_index = np.random.choice(T, k, replace=False)
        self.lam_data[self.contami_index] -= 1.0 * self.G**2
        self.alpha_data = np.ones(T)
        self.alpha = 1.0
        self.alpha_data[self.contami_index] = 0.0

    def value(self, x, t): # function value
        if t in self.contami_index:
            y = 100 * x[0] - 2
        else:
            y = - np.log(0.01+x[0]) + np.log(0.03)
        return y
    
    def grad(self, x, t): # gradient
        y = np.zeros(1)
        if t in self.contami_index:
            y[0] = 100
        else:
            y[0] = -1/(0.01+x[0])
        return y
    
    def hess(self, x, t): # Hessian
        y = np.zeros((1,1))
        if t in self.contami_index:
            pass
        else:
            y[0][0] = 1 / (0.01 + x[0])**2
        return y



# Contaminated strongly convex function
class ContaminatedStronglyConvex:
    def __init__(self, T, k, seed):
        self.T = T # number of rounds
        self.k = k # number of contaminated rounds
        self.D = 1.0 # D in Assumption 3.2
        self.G = 1.0 # G in Assumption 3.3

        # generate contaminated rounds
        self.lam_data = np.ones(T)*1.0
        self.lam = 1.0
        np.random.seed(seed)
        self.contami_index = np.random.choice(T, k, replace=False)
        self.lam_data[self.contami_index] -= 1.0

    def value(self, x, t): # function value
        if t in self.contami_index:
            y = x[0] - (self.T-2.0*self.k) / (self.T-self.k)
        else:
            y = 0.5*(1.0-x[0])**2 - 0.5*(1.0 - (self.T-2.0*self.k) / (self.T-self.k))**2
        return y

    def grad(self, x, t): # gradient
        y = np.zeros(1)
        if t in self.contami_index:
            y[0] = 1.0
        else:
            y[0] = x[0] - 1.0
        return y

    def hess(self, x, t): # Hessian
        y = np.zeros((1,1))
        if t in self.contami_index:
            pass
        else:
            y[0][0] = 1.0
        return y