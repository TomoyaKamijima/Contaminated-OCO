import numpy as np
import cvxpy as cp
import scipy
import math

class Method:
    def __init__(self, obj, T, x0, d):
        self.obj = obj # objective function
        self.T = T # number of rounds
        self.x0 = x0 # initial point
        self.d = d # dimension
        self.lam = self.obj.lam # lambda

    def box_projection(self, x, lower_bound=0, upper_bound=1): # projection onto box
        if x < lower_bound:
            return lower_bound
        elif x > upper_bound:
            return upper_bound
        else:
            return x
    
    def mahalanobis_projection(self, x, A, lower_bound=0, upper_bound=1): # projection onto box based on Mahalanobis distance
        y = cp.Variable(self.d)
        objective = cp.Minimize(cp.quad_form(y - x, A))
        constraints = [y >= lower_bound, y <= upper_bound]
        prob = cp.Problem(objective, constraints)
        prob.solve()
        return y.value

    def GradientDescent(self, k): # OGD
        x = self.x0 # action
        eta_t = [self.obj.D / (self.obj.G*np.sqrt(t+1)) for t in range(self.T)] # stepsize
        sum = 0.0
        regret_data = [] # data of regret
        x_data = [] # data of norm of x
        for t in range(self.T):
            sum += self.obj.value(x, t)
            regret_data.append(sum)
            x_data.append(np.linalg.norm(x))
            x = x - eta_t[t] * self.obj.grad(x, t)
            x = self.mahalanobis_projection(x, np.eye(self.d))
        return regret_data, x_data
    
    def OnlineNewton(self, k): # ONS
        D = self.obj.D # D in Assumption 3.2
        G = self.obj.G # G in Assumption 3.3
        alpha = self.obj.alpha # alpha
        gamma = 0.5 * min(1.0/(G*D), alpha) # gamma
        epsilon = 1.0/(gamma**2 * D**2) # epsilon
        x = self.x0 # action
        A = epsilon * np.eye(self.d) # A
        A_inv = (1.0 / epsilon) * np.eye(self.d) # A^{-1}
        sum = 0.0
        regret_data = [] # data of regret
        x_data = [] # data of norm of x
        for t in range(self.T):
            sum += self.obj.value(x, t)
            regret_data.append(sum)
            x_data.append(np.linalg.norm(x))
            grad = self.obj.grad(x, t)
            A = A + np.dot(grad.reshape(1,-1), grad.reshape(-1,1))
            A_inv = A_inv - (np.dot(np.dot(A_inv, grad.reshape(-1,1)), np.dot(grad.reshape(1,-1), A_inv)))/ (1.0 + np.dot(np.dot(grad.reshape(1,-1), A_inv), grad.reshape(-1,1)))
            x = x - (1.0 / gamma) * np.dot(A_inv, grad)
            if self.d==1: # to avoid error
                x = self.box_projection(x, 0, 1)
            x = self.mahalanobis_projection(x, A)
        return regret_data, x_data
    
    def ContaminatedOGD(self, k): # Contaminated OGD
        D = self.obj.D # D in Assumption 3.2
        G = self.obj.G # G in Assumption 3.3

        # caluculate lambda
        lam_data_sorted = sorted(self.obj.lam_data)
        lam = lam_data_sorted[k] 

        epsilon = G * np.sqrt(2.0) / D # epsilon
        x = self.x0 # action
        A = epsilon * np.eye(self.d) # A
        sum = 0.0
        regret_data = [] # data of regret
        x_data = [] # data of norm of x
        u = 0 # number of contaminated cases
        for t in range(self.T):
            sum += self.obj.value(x, t)
            regret_data.append(sum)
            x_data.append(np.linalg.norm(x))
            if self.obj.lam_data[t]<lam: # contaminated case
                u += 1
                A = A + (G/(D*np.sqrt(2.0*u))) * np.eye(self.d)
            else: # non-contaminated case
                A = A + lam * np.eye(self.d)
            x = x - np.dot(np.linalg.inv(A), self.obj.grad(x, t))
            x = self.mahalanobis_projection(x, A)
        return regret_data, x_data
    
    def ContaminatedONS(self, k): # Contaminated ONS
        D = self.obj.D # D in Assumption 3.2
        G = self.obj.G # G in Assumption 3.3

        # caluculate lambda
        lam_data_sorted = sorted(self.obj.lam_data)
        lam = lam_data_sorted[k]

        alpha = self.obj.alpha # alpha
        gamma = 0.5 * min(1.0/(G*D), alpha) # gamma
        epsilon = G * np.sqrt(2.0) / D # epsilon
        x = self.x0 # action
        A = epsilon * np.eye(self.d) # A
        sum = 0.0
        regret_data = [] # data of regret
        x_data = [] # data of norm of x
        u = 0 # number of contaminated cases
        for t in range(self.T):
            sum += self.obj.value(x, t)
            regret_data.append(sum)
            x_data.append(np.linalg.norm(x))
            if self.obj.lam_data[t]<lam: # contaminated case
                u += 1
                A = A + (G/(D*np.sqrt(2.0*u))) * np.eye(self.d)
            else: # non-contaminated case
                grad = self.obj.grad(x, t)
                A = A + gamma * np.dot(grad.reshape(1,-1), grad.reshape(-1,1))
            x = x - np.dot(np.linalg.inv(A), self.obj.grad(x, t))
            x = self.mahalanobis_projection(x, A)
        return regret_data, x_data
    
    def MetaGrad(self, k): # MetaGrad Master
        D = self.obj.D # D in Assumption 3.2
        G = self.obj.G # G in Assumption 3.3
        I = math.ceil(0.5*math.log2(self.T)) + 1 # maximum number of i
        eta = [2**(-i)/(5*D*G) for i in range(I)] # eta
        C = 1.0 + 1.0 / I # C
        pi = [C/((i+1.0)*(i+2.0)) for i in range(I)] # pi
        w = self.x0 # action
        grad = self.obj.grad(w, 0) # gradient
        sum = 0.0
        regret_data = [] # data of regret
        w_data = [] # data of norm of w
        Sigma = [D**2 * np.eye(self.d) for i in range(I)] # Sigma
        Sigma_inv = [1.0/D**2 * np.eye(self.d) for i in range(I)] # Sigma^{-1}
        w_eta = [np.zeros(self.d) for i in range(I)] # sequence of actions generated by slaves
        for t in range(self.T):
            sum += self.obj.value(w, t)
            regret_data.append(sum)
            w_data.append(np.linalg.norm(w))
            w1 = np.zeros(self.d)
            sum1 = 0.0
            for i in range(I-1, -1, -1):
                Sigma[i], Sigma_inv[i], w_eta[i] = self.MetaGradSlave(eta[i], Sigma[i], Sigma_inv[i], w, w_eta[i], grad)
                w1 = w1 + pi[i] * eta[i] * w_eta[i]
                sum1 += pi[i] * eta[i]
            w = w1 / sum1
            grad = self.obj.grad(w, t)
            l = [- eta[i] * np.dot(w-w_eta[i], grad) + (eta[i] * np.dot(w-w_eta[i], grad))**2 for i in range(I)]
            sum1 = 0.0
            for i in range(I-1, -1, -1):
                sum1 += pi[i] * math.exp(-l[i])
            pi = [pi[i] * math.exp(-l[i]) / sum1 for i in range(I)]
        return regret_data, w_data
    
    def MetaGradSlave(self, eta, Sigma, Sigma_inv, w, w_eta, grad): # MetaGrad Slave
        Sigma = Sigma - 2.0*eta**2*np.dot(np.dot(Sigma, grad).reshape(-1,1), np.dot(grad.reshape(1,-1), Sigma)) / (1.0 + 2.0*eta**2*np.dot(grad, np.dot(Sigma, grad)))
        Sigma_inv = Sigma_inv + 2.0*eta**2*np.dot(grad.reshape(1,-1), grad.reshape(-1,1))
        w_tilde = w_eta - eta * np.dot(Sigma, grad) * (1.0 + 2*eta*np.dot(grad, w_eta - w))
        w = self.mahalanobis_projection(w_tilde, Sigma_inv)
        return Sigma, Sigma_inv, w

