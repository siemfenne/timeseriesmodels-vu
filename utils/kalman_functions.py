import pandas as pd
import numpy as np

def kalman_filter(params, y):
    omega, phi, s2_eta = params
    a0 = omega / (1 - phi)
    P0 = s2_eta / (1 - phi**2) 
    
    a, P = a0, P0
    d = -1.27
    c = omega
    T = phi
    Q = s2_eta
    H = np.pi**2/2

    df_kf = pd.DataFrame(columns=['a_filter', 'P', 'v', 'F', 'K'])

    n_obs = y.shape[0]
    for t in range(n_obs):
        v = y[t] - a - d
        F = P + H
        K = T * P * F**-1

        df_kf.loc[t] = [a, P, v, F, K]

        a = T * a + K * v + c
        P = T**2 * P  + Q -  F * K**2
      
    xi = params[0] / (1 - params[1])
    df_kf['h_filter'] = df_kf['a_filter'] - xi

    return df_kf

def kalman_smoother(params, y):
    df_kf = kalman_filter(params, y)
    df_kf = df_kf.iloc[::-1].reset_index(drop=True)
    df_ks = pd.DataFrame(columns=['r', 'a_smoother', 'N', 'V'])
    r, N = 0, 0
    omega, phi, s2_eta = params

    n_obs = y.shape[0]
    for t in range(n_obs):
        
        r = df_kf['F'][t]**-1 * df_kf['v'][t] + (phi-df_kf['K'][t])*r
        N = df_kf['F'][t]**-1 + (phi-df_kf['K'][t])**2 * N

        a = df_kf['a_filter'][t] + df_kf['P'][t]*r
        V = df_kf['P'][t] - df_kf['P'][t]**2 * N

        df_ks.loc[t] = [r, a, N, V]

    df_ks = df_ks.iloc[::-1].reset_index(drop=True)
    uncon_mean = params[0] / (1 - params[1])
    df_ks['h_smoother'] = df_ks['a_smoother'] - uncon_mean
    return df_ks 

def bootstrap_filter(params, N, y):
    
    omega, phi, s2_eta = params
    n_obs = y.shape[0]
    s2 = s2_eta / (1-phi**2)
    xi = omega / (1-phi)

    df_bs = pd.DataFrame(columns=['a_bootstrap'])

    for t in range(n_obs):
        #step 1
        if t == 0:
            alpha_t = np.random.randn(N) * s2**0.5
        else:
            alpha_t = np.random.randn(N) * s2_eta**0.5 + phi*alpha_t
        #step 2
        weight_t = np.exp(-0.5*(np.log(2*np.pi) + (xi + alpha_t) + (y[t]-y.mean())**2 / np.exp(xi + alpha_t)))
        weight_t /= sum(weight_t)
        #step 3
        a_t = np.dot(weight_t,alpha_t)
        #step 4, contineous resampling
        alpha_t = np.random.choice(alpha_t, size=len(alpha_t), replace=True, p=weight_t)
        df_bs.loc[t] = [a_t]

    return df_bs

def likelihood(params, a0, P0, y, H):
    omega, phi, s2_eta = params
    n_obs = y.shape[0]
    loglik = 0
    a = a0
    P = P0
    d = -1.27
    c = omega
    T = phi
    Q = s2_eta
    for t in range(n_obs):
        v = y[t] - a - d
        F = P + H
        K = T * P * F**-1

        a = T * a + K * v + c
        P = T**2 * P  + Q -  F * K**2

        # Compute the log-likelihood contribution for this time step
        loglik += -0.5 * (np.log(2 * np.pi) + np.log(abs(F)) + v**2 / F)

    return -loglik