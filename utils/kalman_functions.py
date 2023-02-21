import pandas as pd
import numpy as np

def kalman_filter(
    series: pd.Series, 
    a: np.ndarray,
    P: np.ndarray,
    Z: np.ndarray,
    H: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
    Q: np.ndarray,
    )->pd.DataFrame:  
    """ Computes Kalman filter on state space model """
      
    y_list = series.copy().to_list()
    r = pd.DataFrame(columns=['a', 'P', 'v', 'F', 'K', 'ci'])

    for i,y in enumerate(y_list):
        
        # compute all variables
        v = y - Z @ a
        F = Z @ P @ Z.T + H
        K = T @ P @ Z.T @ np.linalg.inv(F)

        # store result for t=i
        r.loc[i] = [a, P, v, F, K, F**.5]
        
        # predict for t+1
        a = T @ a + K @ v
        P = T @ P @ T.T + R @ Q @ R.T - K @ F @ K.T

    r["y"] = series.values
    r.index = series.index
    return r

def kalman_smoother(
    series: pd.Series, 
    r_n: np.ndarray,
    N_n: np.ndarray,
    a: np.ndarray,
    P: np.ndarray,
    Z: np.ndarray,
    H: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
    Q: np.ndarray,
    )->pd.DataFrame:  
    """ Computes Kalman smoother on state space model """
    
    raise Exception("Function not implemented yet!")

    r_f = kalman_filter(series, a, P, Z, H, R, T, Q)
      
    y_list = series.copy().to_list()
    r = pd.DataFrame(columns=['a', 'P', 'v', 'F', 'K', 'ci'])

    # for i,y in enumerate(y_list):
        
    #     # compute all variables
    #     v = y - Z @ a
    #     F = Z @ P @ Z.T + H
    #     K = T @ P @ Z.T @ np.linalg.inv(F)

    #     # store result for t=i
    #     r.loc[i] = [a, P, v, F, K, F**.5]
        
    #     # predict for t+1
    #     a = T @ a + K @ v
    #     P = T @ P @ T.T + R @ Q @ R.T - K @ F @ K.T

    # r["y"] = series.values
    # r.index = series.index
    return r