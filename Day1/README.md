# Torch Hub Series 
## Image Segmentation

Using Pretained model from torch hub that uses FCN model to get real-time segmentation of objects from our inputs.

In this tutorial, you will learn the concept behind Fully Convolutional Networks (FCNs) for segmentation. In addition, we will see how we can use Torch Hub to import a pre-trained FCN model and use it in our projects to get real-time segmentation outputs for our input images.



**Torch Hub Series #6: Image Segmentation**
-------------------------------------------
![](https://github.com/ksajan/100DaysOfLearning/blob/4225e8f6228912f36b88787386357a9511dc8735/Day1/output/segmentation_output.png)

### **Topic Description**

In previous posts of this series, we looked at different computer vision tasks (e.g., classification, localization, depth estimation, etc.), which enabled us to understand the content and its related semantics in images. Furthermore, in a [past tutorial](https://www.pyimagesearch.com/2021/11/08/u-net-training-image-segmentation-models-in-pytorch/), we have developed an in-depth understanding of the image segmentation task and its usefulness in understanding the intricate details and information in images at a pixel level. Additionally, we also looked at segmentation models like UNET, which utilize salient architectural features like skip-connections to effectively segment images in real-time.

In today’s tutorial, we will look at another approach to segment images (i.e., with Fully Convolutional Networks (FCNs)). **Figure 1** shows the high-level architecture of FCN. These networks follow a training paradigm that enables them to effectively segment images using the features learned from computer vision tasks other than segmentation (e.g., classification). Specifically, we will discuss the following in detail:

*   The supervised pre-training paradigm that powers FCN models
*   The architectural modifications that allow FCN models to tackle input images of any size and efficiently compute the segmentation outputs
*   Importing pre-trained FCN Segmentation models from Torch Hub for quick and seamless integration into our deep learning projects
*   Using FCN models with different encoders to segment images in real-time for our own deep-learning applications

![](data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%20700%20357'%3E%3C/svg%3E)

![](https://929687.smushcdn.com/2407837/wp-content/uploads/2022/01/FCN_one-1024x522.png?lossy=1&strip=1&webp=1)

**Figure 1:** Overall architecture of the Fully Convolutional Networks for image segmentation (source: [Long et al.](https://arxiv.org/pdf/1411.4038.pdf)).

### **The FCN Segmentation Model**

Deep learning models trained for different computer vision tasks (e.g., classification, localization, etc.) strive to extract similar features from images to understand image content irrespective of the downstream task at hand. This can be further understood from the fact that attention maps of models trained only for object classification can also point to the location where the particular class object is present in the image, as seen in a [previous tutorial](https://www.pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/). This implies that a classification model has information about both the global object category as well as the position where it is located in the image.

The FCN segmentation model aims to utilize this fact and follows an approach based on repurposing already trained classification models for segmentation. This requires careful design and modifications to the classification model layers to seamlessly convert the model into a segmentation pipeline.

Formally, the FCN approach employs broadly two steps to achieve this. Firstly, it grabs an out-of-the-box model trained on the task of image classification (i.e., ResNet). Next, to convert it into a segmentation model, it replaces the last fully connected layers with the convolutions layer. Note that this can be done by simply using a convolutional layer with kernel size the same as the input feature map dimension (as shown by **Figure 2**).

![](data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%20700%20398'%3E%3C/svg%3E)

![](https://929687.smushcdn.com/2407837/wp-content/uploads/2022/01/FCN_two-1024x582.png?lossy=1&strip=1&webp=1)

**Figure 2:** Converting final fully connected layers to convolution layers (source: [Long et al.](https://arxiv.org/pdf/1411.4038.pdf)).

Since the network is now composed of only convolutional layers without static fully connected layers with a fixed number of nodes, it can take an image of any dimension as input and process it. Furthermore, we append deconvolution layers after the final layers of our modified classification model to map the feature maps back to the original dimension of the input image to get a segmentation output with pixel correspondence with the input.

Now that we have understood the approach behind FCN, let’s go ahead and set up our project directory and see our pre-trained FCN model in action.

### **Configuring Your Development Environment**

To follow this guide, you need to have the PyTorch library, `torchvision` module, and `matplotlib` library installed on your system.

Luckily, these packages are quite easy to install using pip:

```
$ pip install torch torchvision
$ pip install matplotlib
```

### **Project Structure**

We first need to review our project directory structure.

Start by accessing the **_“Downloads”_** section of this tutorial to retrieve the source code and example images.

From there, take a look at the directory structure:

.
├── dataset
│   ├── test\_set
│   └── training\_set
├── output
├── predict.py
└── pyimagesearch
    └── config.py
    └── utils.py

We start by understanding the structure of our project directory. Specifically, the dataset folder stores the `test_set` and the `training_set` images for our Dogs and Cats Dataset. For this tutorial, we will use the `test_set` images for inference and segmentation mask prediction using our FCN model.

The output folder, as usual, stores the visualizations for input images and predicted segmentation masks from our pre-trained FCN model.

Furthermore, the `predict.py` file enables us to load our pre-trained FCN segmentation models from Torch Hub and integrate them into our project for real-time segmentation mask prediction and visualization.

Finally, the `config.py` file in the `pyimagesearch` folder stores our code’s parameters, initial settings, and configurations, and the `utils.py` file defines helper functions that enable us to effectively visualize our segmentation outputs.

### **Downloading the Dataset**

Following the previous tutorials in this series, we’ll use the [Dogs & Cats Images](https://www.kaggle.com/chetankv/dogs-cats-images) dataset from Kaggle. The dataset was introduced as a part of the Dogs vs. Cats image classification challenge and consisted of images belonging to two classes (i.e., Dogs and Cats). The training set comprises 8000 images (i.e., 4000 images for each class), and the test set comprises 2000 images (i.e., 1000 images for each class).

For this tutorial, we will use the test set images from the dataset to conduct inference and generate segmentation masks using our pre-trained FCN model from Torch Hub. The Dogs and Cats dataset is compact and easy to use. Moreover, it consists of the two most common object classes (i.e., dog and cat images) that the classification models in the deep learning community are trained on, making it an apt choice for this tutorial.

### **Creating the Configuration File**

We start by discussing the `config.py` file, which contains the parameter configurations we will use for our tutorial.
```
\# import the necessary packages
import os

# define gpu or cpu usage
DEVICE = "cpu"

# define the root directory followed by the test dataset paths
BASE\_PATH = "dataset"
TEST\_PATH = os.path.join(BASE\_PATH, "test\_set")

#define pre-trained model name and number of classes it was trained on
MODEL = \["fcn\_resnet50", "fcn\_resnet101"\]
NUM\_CLASSES = 21

# specify image size and batch size
IMAGE\_SIZE = 224
BATCH\_SIZE = 4

# define the path to the base output directory
BASE\_OUTPUT = "output"

# define the path to the input image and output segmentation
# mask visualizations
SAVE\_IMAGE\_PATH = os.path.join(BASE\_OUTPUT, "image\_samples.png")
SEGMENTATION\_OUTPUT\_PATH = os.path.sep.join(\[BASE\_OUTPUT,
	"segmentation\_output.png"\])
```
We start by importing the necessary packages on **Line 2** which includes the `os` module for file handling functionalities. Then, on **Line 5**, we define the `DEVICE` that we will use for computation. Note that since we will be using a pre-trained FCN model from Torch Hub for inference, we set the device to CPU as shown.

On **Line 8**, we define the `BASE_PATH` parameter, which points to the location of the root folder where our dataset is stored. Furthermore, we define the `TEST_PATH` parameter on **Line 9**, which points to the location of our test set within the root dataset folder.

Next, we define the `MODEL` parameter, which determines the FCN model that we will use to perform the segmentation task (**Line 12**). Note that Torch Hub provides us access to FCN models with different classification backbones. For example, `fcn_resnet50` corresponds to a pre-trained FCN model with ResNet50 classification backbone and `fcn_resnet101` corresponds to a pre-trained FCN model with ResNet101 classification backbone.

The FCN models hosted on Torch Hub are pre-trained on a subset of COCO train2017, on the 20 categories present in the Pascal Visual Object Classes (VOC) dataset. This makes the total number of categories 21, including the background class. On **Line 13,** we define the total number of classes (i.e., 21) on which the Torch Hub FCN model is pre-trained.

In addition, we define the spatial dimension (i.e., `IMAGE_SIZE`) of the images we will input to our model (**Line 16**) and the `BATCH_SIZE` we will use for loading our image samples (**Line 17**).

Finally, we define the path to our output folder (i.e., `BASE_OUTPUT`) on **Line 20** and the corresponding paths for storing the input image visualizations (i.e., `SAVE_IMG_PATH`) and the final segmentation output plots (i.e., `SEGMENTATION_OUTPUT_PATH`) on **Lines 24 and 25**.

### **Image Segmentation Using FCN Model**

Now that we have defined our parameter configurations, we are ready to set up our project pipeline and see our pre-trained FCN model in action.

As discussed in previous tutorials of this series, Torch Hub provides a simplistic API for accessing models. It can import pre-trained models for various computer vision tasks ranging from classification, localization, depth estimation to generative modeling. Here, we will take a step further and learn to import and use segmentation models from Torch Hub.

Let’s open the `utils.py` file from the pyimagesearch folder in our project directory and start by defining the functions that will help us plot and visualize our segmentation task predictions from our FCN Segmentation model.
```
\# import the necessary packages
from torchvision.utils import draw\_segmentation\_masks
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch

def visualize\_segmentation\_masks(allClassMask, images, numClasses,
	inverseTransforms, device):	
	# convert to boolean masks and batch dimension first format
	booleanMask  = (
		allClassMask == torch.arange(numClasses, device=device)\[:, None, None, None\])
	booleanMask = booleanMask.transpose(1,0)
	# initialize list to store our output masks
	outputMasks = \[\]
	
	# loop over all images and corresponding boolean masks
	for image, mask in zip(images, booleanMask):
		# plot segmentation masks over input images
		outputMasks.append(
			draw\_segmentation\_masks(
				(inverseTransforms(image) \* 255).to(torch.uint8),
				masks=mask,
				alpha=0.6
			)
		)

	# return segmentation plots
	return outputMasks
```
On **Lines 2-6**, we start by importing the necessary packages, which include the `draw_segmentation_masks` function from `torchvision.utils` for segmentation mask visualization (**Line 2**), the functional module from `torchvision.transforms` for image format conversion operations (**Line 3**), and the `matplotlib` library (**Line 4**), to create and visualize our plots, as we will discuss in detail later. Finally, we also import the Numpy and PyTorch libraries for tensor and array manipulations (**Lines 5 and 6**).

We are now ready to define our helper functions to visualize segmentation mask outputs from our FCN segmentation model.

We define the `visualize_segmentation_masks()` function (**Lines 8-29**), which plots the predicted masks over the input images with each class pixel represented with a different color, as we will see later.

The function takes as input the pixel-wise class-level segmentation masks (i.e., `allClassMask`), the input images we want to segment (i.e., `images`), the total number of classes on which our Torch Hub FCN model has been trained (i.e., `numClasses`), the inverse transformations required to convert the image back to unnormalized form (i.e., `inverseTransforms`), and the device we will use for computation (i.e., `device`) as shown on **Lines 8 and 9**.

On **Lines 11 and 12**, we start by creating a Boolean segmentation mask for each of the classes in our dataset, using the input class-level segmentation mask (i.e., `allClassMask`). This corresponds to the `numClasses` number of Boolean masks, which are stored in the `booleanMask` variable (**Line 11**).

The Boolean masks are created using the conditional statement `allClassMask == torch.arange(numClasses, device=device)[:, None, None, None])`.

The left-hand side (LHS) of the statement is simply our class-level segmentation mask with dimension `[batch_size, height=IMG_SIZE,width=IMG_SIZE]`. On the other hand, the right-hand side (RHS) creates a tensor of dim `[numClasses,1,1,1]` with entries `0, 1, 2, …, (numClasses-1)` arranged in sequence.

In order to match the dimensions on both sides of the statement, the RHS tensor is automatically broadcasted spatially (i.e., height and width) to dimension `[numClasses,batch_size,height=IMG_SIZE,width=IMG_SIZE]`. Furthermore, the LHS `allClassMask` is broadcasted on channel dimension to have `numClasses` channels and final shape of `[numClasses,batch_size,height=IMG_SIZE,width=IMG_SIZE]`.

Finally, the conditional statement is executed, which gives us `numClasses` number of Boolean segmentation masks corresponding to each class. The masks are then stored in the `booleanMask` variable.

Note that **Line 12** is a bit complicated to visualize at first. To understand the working of **Line 12**, we start by taking a simplistic example. We will use **Figure 4** to understand the process better.

![](data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%20700%20388'%3E%3C/svg%3E)

![](https://929687.smushcdn.com/2407837/wp-content/uploads/2022/01/understandMasks-1024x568.png?lossy=1&strip=1&webp=1)

**Figure 4:** Understanding Boolean Masks prediction (source: by author).

Let’s say that we have a segmentation task with three classes (class 0, class 1, and class 2) and a single input image in our batch (i.e., `batch_size, bs=1`). As shown, we have a segmentation mask, `S`, which is the predicted output from our segmentation model and represents the LHS of **Line 12** in our example. Notice that `S` is a pixel-wise class-level segmentation mask where each pixel entry corresponds to the class (i.e., 0, 1, or 2) to which the pixel belongs.

On the other hand, the RHS as shown is a tensor of dim `[numClasses = 3,1,1,1]` with entries 0, 1, 2.

The mask `S` is broadcasted on channel dimension to a shape of `[numClasses=3,bs=1,height, width]` as shown. Furthermore, the RHS tensor is broadcasted spatially (i.e., height and width) to have a final shape `[numClasses=3,bs=1,height, width]`.

Notice that the statement’s output is `numClasses=3` number of Boolean segmentation masks corresponding to each class.

We now continue with our explanation of the `visualize_segmentation_masks()` function.

On **Line 13**, we transpose to the `booleanMask` output from **Line 12** to convert it to batch dimension first format with shape `[batch_size,numClasses, height=IMG_SIZE,width=IMG_SIZE]`.

Next, we create an empty list `outputMasks` on **Line 15** to store our final segmentation masks for each image.

On **Line 18**, we start by iterating over all images and their corresponding Boolean masks as shown. For each (`image`, `mask`) pair, we pass the input `image` and `mask` to the `draw_segmentation_masks()` function (**Lines 21-26**).

This function overlays the Boolean masks for each class with a different color over the input `image`. It also takes an alpha parameter which is a value in the range `[0, 1]` and corresponds to the transparency value when the mask is overlaid on the input image (**Line 24**).

Note that the `draw_segmentation_masks()` function expects the input image to be in the range `[0, 255]` and `uint8` format. To achieve this, we use the `inverseTransforms` function to unnormalize our images and convert them to range `[0, 1]` and then multiply by 255 to get the final pixel values in the range `[0, 255]` (**Line 22**). Furthermore, we also set the datatype of the images to `torch.uint8` as expected by our function.

Finally, on **Line 29**, we return the list of output segmentation masks as shown.

Now that we have defined our visualization helper function, we are ready to set up our data pipeline and use our FCN model for the task of image segmentation.

Let’s open the `predict.py` file and get started.
```
\# USAGE
# python predict.py

# import the necessary packages
from pyimagesearch import config
from pyimagesearch import utils
from torchvision.datasets import ImageFolder
from torchvision.utils import save\_image
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import os
```
On **Lines 5-12,** we import the necessary packages and modules, including our `config` file (**Line 5**) and the `utils` module for visualization-based helper functions from the `pyimagesearch` folder (**Line 6**).

We also import the `ImageFolder` class from the `torchvision.datasets` module for creating our dataset (**Line 7**), the `save_image` function for saving our visualization plots (**Line 8**), and the `DataLoader` class from `torch.utils.data` for accessing data-specific functionalities provided by PyTorch to set up our data loading pipeline (**Line 9**).

Finally, we import the `transforms` module from `torchvision` to apply image transformations while loading images (**Line 10**) and the PyTorch and os libraries for tensor and file handling-based functionalities (**Lines 11 and 12**).
```
\# create image transformations and inverse transformation
imageTransforms = transforms.Compose(\[
	transforms.Resize((config.IMAGE\_SIZE, config.IMAGE\_SIZE)),
	transforms.ToTensor(),
	transforms.Normalize(
		mean=\[0.485, 0.456, 0.406\],
		std=\[0.229, 0.224, 0.225\]
	)\])
imageInverseTransforms = transforms.Normalize(
	mean=\[-0.485/0.229, -0.456/0.224, -0.406/0.225\],
	std=\[1/0.229, 1/0.224, 1/0.225\]
)

# initialize dataset and dataloader
print("\[INFO\] creating data pipeline...")
testDs = ImageFolder(config.TEST\_PATH, imageTransforms)
testLoader = DataLoader(testDs, shuffle=True,
	batch\_size=config.BATCH\_SIZE)

# load the pre-trained FCN segmentation model, flash the model to
# the device, and set it to evaluation mode
print("\[INFO\] loading FCN segmentation model from Torch Hub...")
torch.hub.\_validate\_not\_a\_forked\_repo=lambda a,b,c: True
model = torch.hub.load("pytorch/vision:v0.10.0", config.MODEL\[0\],
	pretrained=True)
model.to(config.DEVICE)
model.eval()

# initialize iterator and grab a batch from the dataset
batchIter = iter(testLoader)
print("\[INFO\] getting the test data...")
batch = next(batchIter)

# unpack images and labels and move to device
(images, labels) = (batch\[0\], batch\[1\])
images = images.to(config.DEVICE)

# initialize a empty list to store images
imageList =\[\]

# loop over all images
for image in images:
	# add de-normalized images to the list
	imageList.append(
		(imageInverseTransforms(image) \* 255).to(torch.uint8)
	)
```
Now that we have imported the essential packages, it is time to set up our image transformations.

We define the transformations that we want to apply while loading our input images and consolidate them with the help of the `transforms.Compose` function on **Lines 15-21**. Our `imageTransforms` include:

*   `Resize()`: allows us to resize our images to a particular input dimension (i.e., `config.IMAGE_SIZE`, `config.IMAGE_SIZE`) that our model can accept
*   `ToTensor()`: enables us to convert input images to PyTorch tensors and convert the input PIL Image, which is originally in the range from `[0, 255]`, to `[0, 1]`.
*   `Normalize()`: it takes two arguments, that is, mean and standard deviation (i.e., `mean` and `std`, respectively), and enables us to normalize the images by subtracting the mean and dividing by the given standard deviation. Note that we use the ImageNet statistics for normalizing the images.

Additionally, on **Lines 22-25**, we define an inverse transformation (i.e., `imageInverseTransforms`) which simply performs the opposite operation of the Normalize transformation defined above. This will come in handy when we want to unnormalize the images back to the range `[0, 1]` for visualization.

We are now ready to build our data loading pipeline using PyTorch.

On **Line 29**, we use the `ImageFolder` functionality to create a PyTorch Dataset for our test set images which we will use as input for the segmentation task. `ImageFolder` takes as input the path to our test set (i.e., `config.TEST_PATH`) and the transformation we want to apply to our images (i.e., `imageTransforms`).

On **Lines 30 and 31**, we create our data loader (i.e., `testLoader`) by passing our test dataset (i.e., `testDs`) to the PyTorch `DataLoader` class. We keep the `shuffle` parameter `True` in the data loader since we want to process a different set of shuffled images every time we run our script. Furthermore, we define the `batch_size` parameter using `config.BATCH_SIZE` to determine the number of images in a single batch that the data loader outputs.

Now that we have structured and defined our data loading pipeline, we will initialize our pre-trained FCN segmentation model from Torch Hub.

On **Line 37**, we use the `torch.hub.load` function to load our pre-trained FCN model. Notice that the function takes the following arguments:

*   The location where the model is stored (i.e., `pytorch/vision:v0.10.0`)
*   The name of the model that we intend to load (i.e., `config.MODEL[0]` which corresponds to FCN model with ResNet50 encoder or `config.MODEL[1]` which corresponds to FCN model with ResnNet101 encoder)
*   The `pretrained` parameter, which when set to True, directs the Torch Hub API to download the pre-trained weights of the selected model and load them.

Finally, we transfer our model to `config.DEVICE` using the `to()` function, registering our model and its parameters on the device mentioned (**Line 39**). Since we will be using our pre-trained model for inference, we set the model to `eval()` mode on **Line 40**.

After completing our model definition, it is now time to access samples from our dataLoader and see our Torch Hub FCN segmentation model in action.

We start by converting the `testLoader` iterable to a python iterator using the `iter()` method shown on **Line 43**. This allows us to simply iterate through batches of our dataset with the help of the `next()` method (**Line 45**), as discussed in detail in a [previous tutorial on data loading with PyTorch](https://www.pyimagesearch.com/2021/10/04/image-data-loaders-in-pytorch/).

Since each of the data samples in our batch is a tuple of the form `(images, labels)`, we unpack the test images (i.e., `batch[0]`) and corresponding labels (i.e., `batch[1]`) on **Line 48**. Then, we transfer the batch of images to our model’s device, defined by `config.DEVICE` on **Line 49**.

On **Lines 55-59**, we create an `imageList` by iterating through each image in our `images` tensor and converting it to have pixel values in the range `[0, 255]`. To achieve this, we use the `inverseTransforms` function to unnormalize our images and convert them to range `[0, 1]` and then multiply by 255 to get the final pixel values in the range `[0, 255]`.

Furthermore, we also set the datatype of the images to `torch.uint8` as expected by our visualization functions.
```
\# create the output directory if not already exists
if not os.path.exists(config.BASE\_OUTPUT):
	os.makedirs(config.BASE\_OUTPUT)

# turn off auto grad
with torch.no\_grad():
	# compute prediction from the model
	output = model(images)\["out"\]

# convert predictions to class probabilities
normalizedMasks = torch.nn.functional.softmax(output, dim=1)

# convert to pixel-wise class-level segmentation masks
classMask = normalizedMasks.argmax(1)

# visualize segmentation masks
outputMasks = utils.visualize\_segmentation\_masks(
	allClassMask=classMask,
	images=images,
	numClasses=config.NUM\_CLASSES,
	inverseTransforms=imageInverseTransforms,
	device=config.DEVICE
)

# convert input images and output masks to tensors
inputImages = torch.stack(imageList)
generatedMasks = torch.stack(outputMasks)

# save input image visualizations and the mask visualization
print("\[INFO\] saving the image and mask visualization to disk...")
save\_image(inputImages.float() / 255,
	config.SAVE\_IMAGE\_PATH, nrow=4, scale\_each=True,
	normalize=True)
save\_image(generatedMasks.float() / 255,
	config.SEGMENTATION\_OUTPUT\_PATH, nrow=4, scale\_each=True,
	normalize=True)
```
It is now time to see our pre-trained FCN model in action and use it to generate segmentation masks for input images in our batch.

We first ensure that our output directory where we will be storing the segmentation predictions exists, and if not, we create it as shown on **Lines 62 and 63**.

Since we are only using a pre-trained model for inference, we direct PyTorch to switch off the gradient computation with the help of `torch.no_grad()` as shown on **Line 66**. We then pass our images through our pre-trained FCN model and store the output images in the variable `output` of dimension `[batch_size, config.NUM_CLASSES,config.IMAGE_SIZE,config.IMAGE_SIZE]` (**Line 68**).

As shown on **Line 71**, we convert the predicted segmentation from our FCN model to class probabilities using the `softmax` function. Finally, on **Line 74**, we get the pixel-wise class-level segmentation masks by selecting the most probable class for each pixel position using the `argmax()` function on the class dimension of our `normalizedMasks`.

We then use our `visualize_segmentation_masks()` function from the `utils` module to visualize our `classMask` (**Lines 77-83**) and store the output segmentation masks in the `outputMasks` variable. (**Line 77**).

On **Lines 86 and 87**, we convert our list of input images `imageList` and final `outputMasks` list to tensors by using the `torch.stack()` function to stack the entries in our respective lists. Finally, we use the `save_image` function from `torchvision.utils` to save our `inputImages` tensor (**Lines 91-93**) and `generatedMasks` tensor (**Lines 94-96**).

Note that we convert the tensors to `float()` and normalize them in the range `[0, 1]` by dividing them by `255` for optimal visualization using the `save_image` function.

In addition, we also notice that the `save_image` function takes as input the path where we want to save our images (i.e., `config.SAVE_IMG_PATH` and `config.SEGMENTATION_OUTPUT_PATH`), the number of images displayed in a single row (i.e., `nrow=4`), and two other Boolean parameters, that is, `scale_each` and `normalize` which scale and normalize the image values in the tensor.

Setting these parameters ensures that the images are normalized in the specific ranges required by the `save_image` function for optimal visualization of the results.

**Figure 5** shows the visualization for the input images in our batch (i.e., `inputImages`) on the left and the corresponding predicted segmentation masks from our pre-trained FCN segmentation model on the right (i.e., `generatedMasks`) for four different batches.

![](data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%20700%20411'%3E%3C/svg%3E)

![](https://929687.smushcdn.com/2407837/wp-content/uploads/2022/01/SegmentationPredictions-optimized.png?lossy=1&strip=1&webp=1)

**Figure 5:** Input images and corresponding segmentation outputs (source: by author).

Notice that our FCN model can correctly recognize the pixels corresponding to the category cat (cyan color) and dog (green color) for all cases. Furthermore, we also notice that our model does fairly well even when multiple instances of an object (say, cat in row 2, third image) are present in a single image.

Additionally, we observe that our model can effectively segment our humans (dark blue) in the case where the image contains a human figure and a dog/cat (row 4, second image).

This can be attributed mainly to the 21 categories our FCN model has been pre-trained on, including the classes `cat`, `dog`, and `person`. However, suppose we want to segment an object that is not included in the 21 categories. In that case, we could initialize the FCN model with pre-trained weights from Torch Hub and fine-tune the new classes we want to segment using the transfer learning paradigm.
