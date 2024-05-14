# Welcome ICDE's friends to pay attention to our work :-)
Due to some issues, I cannot attend the on-site conference.   

Since no one explains the poster, I always pay attention to the message. Any questions are welcome!

05.14.2024
--------------------------------------------------------------------------------------
# TGCRN
Time-aware Graph Structure Learning for Spatiao-temporal Forecasting

# Requirements
* numpy >= 1.19.5
* pytorch == 1.2.0
* scipy == 1.4.1

Dependency enviorment can be installed using the following command:

```
pip install -r requirements.txt
```

# Data preparation
The traffic data files for the Shanghai Metro, Hanghzou Metro, NYC-bike, and NYC-taxi are available at [Gooolge Drive](https://drive.google.com/drive/folders/148-iyP8sZ4FtRyfL7SAhH_SOnj7wRCRa?usp=sharing) and [Baidu Drive](https://drive.google.com/drive/folders/148-iyP8sZ4FtRyfL7SAhH_SOnj7wRCRa?usp=sharing). They should be put into data/ corresponding folders. 

# Modeling training
```
cd ./model 

# HZMetro 
python run.py --dataset ../data/HZMetro --data HZ --lag 4 --horizon 4 --num_nodes 80

# SHMetro
python run.py --dataset ../data/SHMetro --data SH --lag 4 --horizon 4 --num_nodes 288

# Taxi
python run.py --dataset ../data/taxi --data taxi --lag 12 --horizon 12 --num_nodes 266

# Bike
python run.py --dataset ../data/bike --data bike --lag 12 --horizon 12 --num_nodes 250
```

# Cite
Please cite our work if you find it useful.

```
@inproceedings{ma2024tgcrn,
  title={Learning Time-aware Graph Structures for Spatially Correlated Time Series Forecasting},
  author={Ma, Minbo and Hu, Jilin and Jensen, Christian S and Teng, Fei and Han, Peng and Xu, Zhiqiang and Li, Tianrui},
  booktitle={ICDE},
  year={2024}
}
```
