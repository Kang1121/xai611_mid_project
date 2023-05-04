# xai611_mid_project

### Platform
The code is tested on Windows 11 with Python 3.8.16 and Pytorch 2.0.0, some dependencies may differ on other platforms.

### Clone
Open terminal
```
git clone git@github.com:Kang1121/xai611_mid_project.git
```

### Dataset
Download the [preprocessed dataset](https://drive.google.com/file/d/10E9swE6tPHYi3G5mFXwUuuwuOBWh6_rI/view?usp=sharing
) and put it in 'data' folder.


### Create the enviroment
```
cd xai611_mid_project
conda env create -f environment.yml
conda activate xai611
pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git
```
### Run in the terminal
```
python main.py -dataset bcic_iv_2a -model EEGNetv4 -proportion 0.3 -random_pick
```

### Run in batch
```
source run_script.sh
```
