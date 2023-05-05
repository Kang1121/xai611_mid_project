@echo off 
conda activate xai611 & python main.py -dataset bcic_iv_2a -model EEGNetv4 -proportion 1 -fold 5 &&
start  /wait conda activate xai611 && python main.py -dataset bcic_iv_2a -model EEGNetv4 -proportion 0.3 -random_pick -fold 5
start  /wait conda activate xai611 && python main.py -dataset bcic_iv_2a -model EEGNetv4 -proportion 0.6 -random_pick -fold 5
start  /wait conda activate xai611 && python main.py -dataset bcic_iv_2a -model EEGNetv4 -proportion 0.8 -random_pick -fold 5
pause