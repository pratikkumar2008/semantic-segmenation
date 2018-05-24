from sklearn.datasets import load_files
from tqdm import tqdm
from PIL import Image
import numpy as np
import copy

       

def load_dataset(path):
    data = load_files(path,load_content=False)
    return data.filenames
	
def path_to_tensor_RGB(img_path):
    print (img_path)
    img = Image.open(img_path)
	#resizing images to 320*160
    img = img.resize((320,160),Image.ANTIALIAS)
    x = np.asarray(img)
    return np.expand_dims(x, axis=0)


def paths_to_tensor_RGB(img_paths):
    list_of_tensors = [path_to_tensor_RGB(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)


def path_to_tensor_RGB_One_hot_encoding(img_path):
    #print (img_path)
    img = Image.open(img_path)
	#resizing segmented images to 320*160
    img = img.resize((320,160),Image.ANTIALIAS)
    x = np.asarray(img)
    y=copy.copy(x)
    x=None
    np.place(y, y==255, [0])
    y=(np.arange(20+1) == y[...,None]).astype(int)
    print(y.shape)
    return np.expand_dims(y, axis=0)


def paths_to_tensor_RGB_One_hot_encoding(img_paths):
    list_of_tensors = [path_to_tensor_RGB_One_hot_encoding(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

