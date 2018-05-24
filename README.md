Software requirements:
1. python 3.x
2. numpy
3. pillow
4. Keras
5. tqdm
6. tensorflow
7. matplotlib
8. jupyter notebook
(Preferably install using anaconda)

User-defined Modules -
1. loadData.py
2. myaccuracy.py
3. myModels.py

Jupyter notebook-
1. udacity_final.ipynb

Model architectures:
1. model_seq.png
2. model_skip.png




Dataset can be downloaded from  http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit

Classes to be detected in dataset with the corresponding pixel values.
'background'-0
'aeroplane'-1
'bicycle' 2 
'bird' 3
'boat' 4
'bottle' 5
'bus' 6
'car' 7
'cat' 8
'chair' 9
'cow' 10
'diningtable' 11
'dog' 12 
'horse' 13
'motorbike' 14
'person' 15
'potted-plant' 16
'sheep' 17
'sofa' 18
'train' 19
'tv/monitor' 20
'ambigious' (transformed to 0)


Directories to be created
1. saved_models
2. data
3. data/trainInput
4. data/trainInput/trainInput --> place all input images here.
5. data/trainClass
6. data/trainClass/trainClass --> place all segmented images here.
