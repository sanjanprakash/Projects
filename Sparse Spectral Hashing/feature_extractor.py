
# coding: utf-8

#Predict and Extract Features with Pre-trained Models
# 
#We download a pre-trained Resnet 50-layer model on Imagenet. Other models are available at http://data.mxnet.io/models/

# In[3]:

import os, urllib
def download(url):
    filename = url.split("/")[-1]
    if not os.path.exists(filename):
        urllib.urlretrieve(url, filename)
def get_model(prefix, epoch):
    download(prefix+'-symbol.json')
    download(prefix+'-%04d.params' % (epoch,))

get_model('http://data.mxnet.io/models/imagenet/resnet/50-layers/resnet-50', 0)


#Initialization
#We first load the model into memory with `load_checkpoint`. It returns the symbol definition of the neural network, and parameters. 

# In[4]:

import mxnet as mx
sym, arg_params, aux_params = mx.model.load_checkpoint('resnet-50', 0)


#We can visualize the neural network by `mx.viz.plot_network`. Uncomment to see the network

# In[5]:

#mx.viz.plot_network(sym)


#We create an executable `module`. Context tells about the device `mx.cpu()` for CPU and `mx.gpu()` for the  GPU. 

# In[6]:

mod = mx.mod.Module(symbol = sym, context = mx.cpu())


#The ResNet is trained with RGB images of size 224 x 224. The training data is fed by the variable `data`. We bind the module with the input shape and specify that it is only for predicting. The number 1 added before the image shape (3x224x224) means that we will only predict one image each time. Next, we set the loaded parameters. Now the module is ready to run. 

# In[7]:

mod.bind(for_training = False, data_shapes = [('data', (1,3,224,224))])
mod.set_params(arg_params, aux_params)


#Prepare data
#We first obtain the synset file, in which the i-th line contains the label for the i-th class

# In[8]:

download('http://data.mxnet.io/models/imagenet/resnet/synset.txt')
with open('synset.txt') as f:
    synsets = [l.rstrip() for l in f]


#We next download 1000 images for testing, which were not used for the training 

# In[9]:

import tarfile
download('http://data.mxnet.io/data/val_1000.tar')
tfile = tarfile.open('val_1000.tar')
tfile.extractall()
with open('val_1000/label') as f:
    val_label = [int(l.split('\t')[0]) for l in f]


#Visualize the first 8 images.

# In[8]:

get_ipython().magic(u'matplotlib inline')
import matplotlib
matplotlib.rc("savefig", dpi = 100)
import matplotlib.pyplot as plt
import cv2
for i in range(0,8):
    img = cv2.cvtColor(cv2.imread('val_1000/%d.jpg' % (i,)), cv2.COLOR_BGR2RGB)
    plt.subplot(2,4,i+1)
    plt.imshow(img)
    plt.axis('off')
    label = synsets[val_label[i]]
    label = ' '.join(label.split(',')[0].split(' ')[1:])
    plt.title(label)


#Next, we define a function that reads one image each time and convert to a format that can be used by the model. Here, we use a naive way that resizes the original image into the desired shape, and change the data layout. 

# In[10]:

import numpy as np
import cv2
def get_image(filename):
    img = cv2.imread(filename)  #Read image in BGR order
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   #Change to RGB order
    img = cv2.resize(img, (224, 224))  #Resize to 224*224 to fit model
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)  #Change to (channel, height, width)
    img = img[np.newaxis, :]  #Extend to (example, channel, heigth, width)
    return img


#Finally, we define an input data structure which is acceptable by mxnet. The field `data` is used for the input data, which is a list of NDArrays.

# In[11]:

from collections import namedtuple
Batch = namedtuple('Batch', ['data'])


#Predict
#Now, we are ready to run the prediction by `forward`. Then, we can get the output using `get_outputs`, in which the i-th element is the predicted probability that the image contains the i-th class. 

# In[12]:

img = get_image('val_1000/0.jpg')
mod.forward(Batch([mx.nd.array(img)]))
prob = mod.get_outputs()[0].asnumpy()
y = np.argsort(np.squeeze(prob))[::-1]
print('truth label %d; top-1 predict label %d' % (val_label[0], y[0]))


#When predicting more than one image, we can batch several images together which potentially improves the performance

# In[13]:

batch_size = 32
mod2 = mx.mod.Module(symbol = sym, context = mx.cpu())
mod2.bind(for_training = False, data_shapes = [('data', (batch_size,3,224,224))])
mod2.set_params(arg_params, aux_params)


#Now, we iterate multiple images to calculate the accuracy

# In[14]:

#Output may vary
import time
acc = 0.0
total = 0.0
for i in range(0, 200/batch_size):
    tic = time.time()
    idx = range(i * batch_size, (i + 1) * batch_size)
    img = np.concatenate([get_image('val_1000/%d.jpg' % (j)) for j in idx])
    mod2.forward(Batch([mx.nd.array(img)]))
    prob = mod2.get_outputs()[0].asnumpy()
    pred = np.argsort(prob, axis=1)
    top1 = pred[:,-1]
    acc += sum(top1 == np.array([val_label[j] for j in idx]))
    total += len(idx)
    print('batch %d, time %f sec'%(i, time.time() - tic))
assert acc/total > 0.66, "Low top-1 accuracy."
print('top-1 accuracy %f'%(acc/total))


#Extract Features
#The neural network works as a feature extraction module for other applications 
#A loaded symbol in default only returns the last layer as output. But we can get all internal layers by `get_internals`, which returns a new symbol outputting all internal layers. The following codes print the last 10 layer names. 
 
#We can also use `mx.viz.plot_network(sym)` to visually find the name of the layer we want to use. The name conventions of the output is the layer name with `_output` as the postfix.

# In[15]:

all_layers = sym.get_internals()
all_layers.list_outputs()[-10:-1]


#Often we want to use the output before the last fully connected layers, which may return semantic features of the raw images but not too fitting to the label yet. In the ResNet case, it is the flatten layer with name `flatten0` before the last fullc layer. The following codes get the new symbol `sym3` which use the flatten layer as the last output layer, and initialize a new module.

# In[16]:

all_layers = sym.get_internals()
sym3 = all_layers['flatten0_output']
mod3 = mx.mod.Module(symbol = sym3, context = mx.cpu())
mod3.bind(for_training = False, data_shapes = [('data', (1,3,224,224))])
mod3.set_params(arg_params, aux_params)


#Now we can do feature extraction using `forward1` as before. Notice that the last convolution layer uses 2048 channels, and we then perform an average pooling, so the output size of the flatten layer is 2048.

#We now add the classes we want to classify on.

# In[19]:

#Extraction of features for our classes
from PIL import Image
import glob

out = []
data_pos = np.array([get_image('Dataset/football/' + img) for img in os.listdir('Dataset/football/')])
data_neg = np.array([get_image('Dataset/lion/' + img) for img in os.listdir('Dataset/lion/')])
# data_pos1 = np.array([get_image('Dataset/piano/' + img) for img in os.listdir('Dataset/piano/')])
data_neg1 = np.array([get_image('Dataset/guitar/' + img) for img in os.listdir('Dataset/guitar/')])
datas = np.array([get_image('Dataset/oxford/' + img) for img in os.listdir('Dataset/oxford/')])

X = np.append(data_pos,data_neg,axis = 0)
Y = np.append(data_neg1,datas,axis = 0)
Z = np.append(X,Y,axis = 0)

print Z, Z.shape

import pandas


# pd = pandas.DataFrame(Z.reshape(1,-1))
# pd.to_csv("imagelist.csv")
  


# In[40]:


for img in Z:
    mod3.forward(Batch([mx.nd.array(img)]))
    out.append(mod3.get_outputs()[0].asnumpy())
out = np.array(out)


#We feed the features into Decision tree classifier

# In[48]:

print out,out.shape
# import csv

import pandas

pd = pandas.DataFrame(out)
pd.to_csv("mylist.csv")
