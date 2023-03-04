import matplotlib.pyplot as plt
import numpy as np
x = [1000,10000,100000,1000000]
x = np.log10(x)
serial_opt_y=[0.0172102,0.202334,2.1064,52.3385]
serial_opt_y_log=np.log10(serial_opt_y)
plt.plot(x, serial_opt_y_log, color='green', marker='o', label="serial_opt_y",linestyle='dashed',linewidth=2, markersize=12)
openmp_opt_y=[0.00999202,0.0635597,0.569275,5.30763]
openmp_opt_y_log=np.log10(openmp_opt_y)
plt.plot(x, openmp_opt_y_log, color='red', marker='o' ,label="openmp_opt_y", linestyle='dashed',linewidth=2, markersize=12)

plt.title("Performance of OpenMP vs Serial O(n) Algorithm")
plt.xlabel("Log Number of Particles")
plt.ylabel("Log Simulation Time (s)")
plt.legend()

plt.tight_layout()
plt.show()

plt.savefig('plot.png')
