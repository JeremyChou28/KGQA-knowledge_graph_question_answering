cd ..
cd src
nohup python run_similarity.py > ../log/webqa/sim/webqa_similarity.log 2>&1 &

jobs -l
