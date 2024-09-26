import numpy as np

threshold = 20 * np.pi / 180
theta = -110 * np.pi / 180

print(theta)

print((np.abs(theta)//threshold)*threshold* np.sign(theta))



# 0 till 0.1      0
# 0.1 till 0.2    0.1
# 0.2 till 0.3    0.2