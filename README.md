AWSGPU_DeepLearning
===================

Code to setup AWS GPU instance to run Daniel Nouri's Facial Keypoints competition


1. start a AWS GPU instance
 - I used Ubuntu Server 14.04 LTS (HVM), SSD Volume Type
 - Search "community AMI's" for ```ami-1f3e225a```
 - And for GPU instance I used GPU instances g2.2xlarge
2. ssh into the server
3. ```git clone https://github.com/wendykan/AWSGPU_DeepLearning.git```
4. ```chmod 777 -R AWSGPU_DeepLearning/```
5. ```vi cookies.txt```
  - Then copy-paste kaggle cookie into this file
6. ```./AWSGPU_DeepLearning/setup.sh ```
7. ```python AWSGPU_DeepLearning/kfkd.py```
