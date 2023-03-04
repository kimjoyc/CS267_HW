import matplotlib.pyplot as plt
import numpy as np
x = [1000,10000,100000,1000000]
x = np.log10(x)

openmp_thread_8=[0.0049782,0.0364074,0.320708,10.1086]
openmp_thread_8_log=np.log10(openmp_thread_8)
plt.plot(x, openmp_thread_8_log, color='green', marker='o', label="openmp_thread_8",linestyle='dashed',linewidth=2, markersize=12)

openmp_thread_16=[0.00967882,0.045685,0.399801,7.07425]
openmp_thread_16_log=np.log10(openmp_thread_16)
plt.plot(x, openmp_thread_16_log, color='red', marker='o' ,label="openmp_thread_16", linestyle='dashed',linewidth=2, markersize=12)


openmp_thread_24=[0.00885058,0.0462823,0.399105,6.3474]
openmp_thread_24_log=np.log10(openmp_thread_24)
plt.plot(x, openmp_thread_24_log, color='blue', marker='o' ,label="openmp_thread_24", linestyle='dashed',linewidth=2, markersize=12)


openmp_thread_32=[0.00858812,0.0497697,0.413151,5.48306]
openmp_thread_32_log=np.log10(openmp_thread_32)
plt.plot(x, openmp_thread_32_log, color='purple', marker='o' ,label="openmp_thread_32", linestyle='dashed',linewidth=2, markersize=12)




plt.title("Performance of OpenMP Pthreads vs Particle Size")
plt.xlabel("Log Number of Particles")
plt.ylabel("Log Simulation Time (s)")
plt.legend()


plt.tight_layout()
plt.show()

plt.savefig('plot2.png')
