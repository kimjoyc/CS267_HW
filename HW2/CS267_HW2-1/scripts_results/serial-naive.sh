#run the application:
num="1000 10000 100000 1000000"
for i in $num; 
do
   # ./serial -n $i >> serial-naive.out
   ./serial -n $i 

done
