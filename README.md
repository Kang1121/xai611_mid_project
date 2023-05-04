# xai611_mid_project

### Clone
Open terminal
```
git clone git@github.com:Kang1121/xai611_mid_project.git
```
### Create the enviroment
```
cd xai611_mid_project
conda env create -f environment.yml
pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git
```
### Run a single program
create "data", "results" and "checkpoints" three folders by yourself, and put data under folder "data"
```
python main.py
```

### Run with shell
```
source run_script.sh
```
