AWSGPU_DeepLearning
===================

Code to setup AWS GPU instance to run Daniel Nouri's Facial Keypoints competition


1. start a AWS GPU instance
 - I used Ubuntu Server 14.04 LTS (HVM), SSD Volume Type 
 - And for GPU instance I used GPU instances g2.2xlarge
2. ssh into the server
3. ```sudo apt-get install git```
4. ```git clone https://github.com/wendykan/AWSGPU_DeepLearning.git```
5. ```chmod 777 -R AWSGPU_DeepLearning/```
6. ```vi cookies.txt```
  - Then copy-paste kaggle cookie into this file
7. ```./AWSGPU_DeepLearning/setup.sh ```
8. ```python AWSGPU_DeepLearning/kfkd.py```
