cd models
cat best_part* > best.h5
cd ../
python3 ensemble.py $1 $2
