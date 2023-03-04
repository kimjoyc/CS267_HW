#run the application:
num="1000 10000 100000 1000000"
for i in $num; 
do
   # ./serial-opt -n $i >> serial-opt.out
    ./serial-opt -n $i

done
