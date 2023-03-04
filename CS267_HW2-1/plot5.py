import matplotlib.pyplot as plt
import numpy as np
x = [8,16,24,32]
x = np.log10(x)

openmp_prob_size_1000=[0.0049782,0.00967882,0.00885058,0.00858812]
openmp_prob_size_1000_log=np.log10(openmp_prob_size_1000)
plt.plot(x, openmp_prob_size_1000_log, color='green', marker='o', label="openmp_prob_size_1000",linestyle='dashed',linewidth=2, markersize=12)

openmp_prob_size_10000=[0.0364074,0.045685,0.0462823,0.0497697]
openmp_prob_size_10000_log=np.log10(openmp_prob_size_1000)
plt.plot(x, openmp_prob_size_10000_log, color='red', marker='o' ,label="openmp_prob_size_10000", linestyle='dashed',linewidth=2, markersize=12)


openmp_prob_size_100000=[0.320708,0.399801,0.399105,0.413151]
openmp_prob_size_100000_log=np.log10(openmp_prob_size_100000)
plt.plot(x, openmp_prob_size_100000_log, color='blue', marker='o' ,label="openmp_prob_size_100000", linestyle='dashed',linewidth=2, markersize=12)


openmp_prob_size_1000000=[10.1086,7.07425,6.3474,5.48306]
openmp_prob_size_1000000_log=np.log10(openmp_prob_size_1000000)
plt.plot(x, openmp_prob_size_1000000_log, color='purple', marker='o' ,label="openmp_prob_size_1000000", linestyle='dashed',linewidth=2, markersize=12)




plt.title("Performance of OpenMP Pthreads vs Particle Size")
plt.xlabel("Log Number of Particles")
plt.ylabel("Log Simulation Time (s)")
plt.legend()


plt.tight_layout()
plt.show()

plt.savefig('plot2.png')
