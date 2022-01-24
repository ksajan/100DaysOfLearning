import os

DEVICE = "cpu"

BASE_PATH = "dataset"
TEST_PATH = os.path.join(BASE_PATH, "test_set")

#define pre-trained model name and number of classes it was trained on
MODEL = ["fcn_resnet50", "fcn_resnet101"]
NUM_CLASSES = 21
# specify image size and batch size
IMAGE_SIZE = 224
BATCH_SIZE = 2
# define the path to the base output directory
BASE_OUTPUT = "output"
# define the path to the input image and output segmentation
# mask visualizations
SAVE_IMAGE_PATH = os.path.join(BASE_OUTPUT, "image_samples.png")
SEGMENTATION_OUTPUT_PATH = os.path.sep.join([BASE_OUTPUT, "segmentation_output.png"])
