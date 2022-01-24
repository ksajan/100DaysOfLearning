# import the necessary packages
from torchvision.utils import draw_segmentation_masks
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch


def visualize_segmentation_masks(allClassMask, images, numClasses,
                                inverseTransforms, device):
	# convert to boolean masks and batch dimension first format
	booleanMask = (
		allClassMask == torch.arange(numClasses, device=device)[:, None, None, None])
	booleanMask = booleanMask.transpose(1, 0)
	# initialize list to store our output masks
	outputMasks = []

	# loop over all images and corresponding boolean masks
	for image, mask in zip(images, booleanMask):
		# plot segmentation masks over input images
		outputMasks.append(
			draw_segmentation_masks(
				(inverseTransforms(image) * 255).to(torch.uint8),
				masks=mask,
				alpha=0.6
			)
		)
	# return segmentation plots
	return outputMasks
