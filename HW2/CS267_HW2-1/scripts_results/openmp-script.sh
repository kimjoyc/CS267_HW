
#OpenMP settings:
export OMP_NUM_THREADS=64
export OMP_PLACES=cores
#run the application:
num="1000 10000 100000 1000000"
for i in $num; 
do
   ./openmp -n $i
done
