# MSGFN-master
Source code of MSGFN framework and comparisons.
1. Code Description
data: raw mdd and hc data
dsc: definitation of the dsc model
gcn: definitation of the gcn model
pretrained_model: saved dsc model and gcn model 
pre_main.py: test the performance of saved models
Run pre_main.py to test the performance of the proposed method.
2. Requirements
python 3.6
tensorflow-gpu 1.12.0

##  Multi-Stage Graph Fusion Networks for Major Depressive Disorder Diagnosis

Youyong Kong, Shuyi Niu, Heren Gao, Yingying Yue, Huazhong Shu, Chunming Xie, Zhijun Zhang, Yonggui Yuan

###  Model Architecture

![image](https://user-images.githubusercontent.com/88756798/175452145-2b42de8d-df04-4d26-a231-72d34140bb27.png)

###   Code Description

Our code can be found in the Model-run folder.

- dsc: definitation of the dsc model
- gcn: definitation of the gcn model
- pretrained_model: saved dsc model and gcn model 
- pre_main.py: test the performance of saved models

Run pre_main.py to test the performance of the proposed method.

### Environment
The code is developed and tested under the following environment

- Python 3.6
- Tensorflow 1.12.0

### Comparisons

To ensure the fairness of the results, we all use source code or public libraries for comparison experiments.  
The results can be obtained from the following documents:  
 
 - GCNII：/Comparisons/GCNII-master/train.py  
 - APPNP：/Comparisons/APPNP/test_pytorch.py  
 - SimpGCN:SimpGCN.py（import DeepRobust） 
