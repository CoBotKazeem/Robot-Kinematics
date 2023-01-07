import numpy as np
import modern_robotics as mr
import IKinBodyIterates as ik

L1 = .425
L2 = .392
W1 = .109
W2 = .082
H1 = .089
H2 = .095

Blist = np.array([	[    0,      0,    0,  0,   0, 0],
					[    1,      0,    0,  0,  -1, 0],
					[    0,      1,    1,  1,   0, 1],
					[W1+W2,     H2,   H2, H2, -W2, 0],
					[    0, -L1-L2,  -L2,  0,   0, 0],
					[L1+L2,      0,    0,  0,   0, 0]])

M = np.array([[-1,0,0,L1+L2],[0,0,1,W1+W2],[0,1,0,H1-H2],[0,0,0,1]])

Tsd = np.array([[ 0, 1,  0, -0.5],
				[ 0, 0, -1,  0.1],
				[-1, 0,  0,  0.1],
				[ 0, 0,  0,    1]])

thetalist0 = np.array([ 2.76, -0.89, 1.36, -1.07, -0.2, 4.72])

sol,boo = ik.IKinBodyIterates(Blist, M, Tsd, thetalist0, 0.001, 0.0001)

print(sol)