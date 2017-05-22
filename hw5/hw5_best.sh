cd models
wget https://github.com/haochunchang/ML2017/releases/download/v.0.51172-public/best_model.json
wget https://github.com/haochunchang/ML2017/releases/download/v.0.51172-public/best.h5
cd ../
python3.5 ensemble.py $1 $2
