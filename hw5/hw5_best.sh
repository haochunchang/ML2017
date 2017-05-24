cd models
cat best* > best.h5
cd ../
python3 ensemble.py $1 $2
