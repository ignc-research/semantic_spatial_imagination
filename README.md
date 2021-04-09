# semantic_spatial_imagination
This repository contains a PyTorch implementation of our IROS-21 paper: Spatial Imagination With Semantic Cognition for Mobile Robots 

# Abstract
The imagination of the surrounding environment based on the experience and semantic cognition has great potential to extend the limited observations and provide more information for mapping, collision avoidance and path planning. This paper provides a training-based algorithm for mobile robots to perform spatial imagination based on semantic cognition and evaluates the proposed method for the mapping task. We utilize a photo-realistic simulation environment, Habitat, for training and evaluation. The trained model is composed of Resent-18 as encoder and Unet as the backbone. We demonstrate that the algorithm can perform imagination for unseen parts of the object universally, by recalling the images and experience and compare our approach with traditional semantic mapping methods. It is found that our approach will improve the efficiency and accuracy of semantic mapping. 
# Install Habitat
```
export ROOT_DIR=$PWD
git submodule init 
git submodule update
cd $ROOT_DIR/environments/habitat/habitat-api
python setup.py develop --all
```

```
cd $ROOT_DIR/environments/habitat/habitat-sim
python setup.py install --headless --with-cuda
```

# Install Other Packages
install required python packages
```
pip install -r ./requirements.txt
```

install pytorch and torchvision
```
pip install --upgrade numpy
onda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch
```

# Run Training

```
cd $ROOT_DIR
export $PYTHONPATH=$PWD
python ./tools/semantic_anti_train/semantic_anti_trainner.py
```

# Run Evaluation
```
cd $ROOT_DIR
export $PYTHONPATH=$PWD
python ./tools/semantic_anti_eval/semantic_anti_eval.py
```

# Citation
```

```
