# USAGE
# python predict.py
# import the necessary packages
from src import config
from src import utils
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import os


# create image transformations and inverse transformation
imageTransforms = transforms.Compose([
	transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
	transforms.ToTensor(),
	transforms.Normalize(
		mean=[0.485, 0.456, 0.406],
		std=[0.229, 0.224, 0.225]
	)])
imageInverseTransforms = transforms.Normalize(
	mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
	std=[1/0.229, 1/0.224, 1/0.225]
)
# initialize dataset and dataloader
print("[INFO] creating data pipeline...")
testDs = ImageFolder(config.TEST_PATH, imageTransforms)
testLoader = DataLoader(testDs, shuffle=False,
                        batch_size=config.BATCH_SIZE)
# load the pre-trained FCN segmentation model, flash the model to
# the device, and set it to evaluation mode
print("[INFO] loading FCN segmentation model from Torch Hub...")
torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
model = torch.hub.load("pytorch/vision:v0.10.0", config.MODEL[0],
                    pretrained=True)
model.to(config.DEVICE)
model.eval()
# initialize iterator and grab a batch from the dataset
batchIter = iter(testLoader)
print("[INFO] getting the test data...")
batch = next(batchIter)
# unpack images and labels and move to device
(images, labels) = (batch[0], batch[1])
images = images.to(config.DEVICE)
# initialize a empty list to store images
imageList = []
# loop over all images
for image in images:
	# add de-normalized images to the list
	imageList.append(
		(imageInverseTransforms(image) * 255).to(torch.uint8)
	)

# create the output directory if not already exists
if not os.path.exists(config.BASE_OUTPUT):
	os.makedirs(config.BASE_OUTPUT)
# turn off auto grad
with torch.no_grad():
	# compute prediction from the model
	output = model(images)["out"]
# convert predictions to class probabilities
normalizedMasks = torch.nn.functional.softmax(output, dim=1)
# convert to pixel-wise class-level segmentation masks
classMask = normalizedMasks.argmax(1)
# visualize segmentation masks
outputMasks = utils.visualize_segmentation_masks(
	allClassMask=classMask,
	images=images,
	numClasses=config.NUM_CLASSES,
	inverseTransforms=imageInverseTransforms,
	device=config.DEVICE
)
# convert input images and output masks to tensors
inputImages = torch.stack(imageList)
generatedMasks = torch.stack(outputMasks)
# save input image visualizations and the mask visualization
print("[INFO] saving the image and mask visualization to disk...")
save_image(inputImages.float() / 255,
           config.SAVE_IMAGE_PATH, nrow=4, scale_each=True,
           normalize=True)
save_image(generatedMasks.float() / 255,
           config.SEGMENTATION_OUTPUT_PATH, nrow=4, scale_each=True,
           normalize=True)
