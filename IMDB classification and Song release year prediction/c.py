import numpy as np
import pandas as pd
import sys

x_train = pd.read_csv(sys.argv[1], header=None)

x_test = pd.read_csv(sys.argv[2], header=None)
x_train = x_train.values
x_test = x_test.values
y_test = list(x_test[:, 0])
y_train = list(x_train[:, 0])
x_test[:, 0] = 1
x_train[:, 0] = 1

reg = lm.LassoLars(alpha=10 ** (-5), normalize=True, fit_intercept=True, precompute='auto', max_iter=1000, eps=0.1, copy_X=True, fit_path=True, positive=False)
reg.fit(x_train, y_train)
x = reg.predict(x_test)

np.savetxt(sys.argv[3], x)

