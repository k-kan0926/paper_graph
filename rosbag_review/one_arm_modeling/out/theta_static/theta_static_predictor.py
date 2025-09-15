# auto-generated
import numpy as np
coef = np.array([0.06639730442331124, 0.3804787689468642, 0.36595176818952185, -0.6924655850815562, -0.6181253059242947, 2.3907398818967187, -0.7776819847483032, 1.5472606292283968, -2.2158613083053265, -0.9522158980255435], dtype=float)
deg = 3
def theta_hat(ps, pd):
    # ps, pd: numpy配列 or スカラ(MPa) -> theta(rad)
    ps = np.asarray(ps); pd = np.asarray(pd)
    feats=[]
    for i in range(deg+1):
        for j in range(deg+1-i):
            feats.append((ps**i)*(pd**j))
    X = np.column_stack(feats) if np.ndim(ps)>0 or np.ndim(pd)>0 else np.array(feats).reshape(1,-1)
    return X @ np.asarray(coef)
