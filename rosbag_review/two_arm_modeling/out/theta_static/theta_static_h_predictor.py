# auto-generated
import numpy as np
coef = np.array([-0.03322215793953045, 1.0832791768109384, -0.03564301273929334, -0.68772780576854, -0.18571801979133237, 0.2930979939289299, 0.09380963292689735, 0.43761896347878865, -0.7126882393130278, -0.2483580488596873], dtype=float)
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
