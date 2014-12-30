sudo apt-get -y update
sudo apt-get -y upgrade
sudo apt-get -y dist-upgrade
sudo apt-get -y install git make python-dev python-setuptools libblas-dev gfortran g++ python-pip python-numpy python-scipy liblapack-dev
sudo pip install ipython nose
sudo pip install --upgrade git+git://github.com/Theano/Theano.git
sudo pip install --upgrade theano
sudo pip install -r https://raw.githubusercontent.com/dnouri/kfkd-tutorial/master/requirements.txt
sudo pip install -r https://raw.githubusercontent.com/dnouri/kfkd-tutorial/master/requirements-2.txt
 
# start running things here
wget https://www.kaggle.com/c/facial-keypoints-detection/download/training.zip
wget https://www.kaggle.com/c/facial-keypoints-detection/download/test.zip
wget https://www.kaggle.com/c/facial-keypoints-detection/download/SampleSubmission.csv
wget https://www.kaggle.com/c/facial-keypoints-detection/download/IdLookupTable.csv
unzip training.zip
unzip test.zip
 
