from mrcnn.config import Config

class MaskRCNNConfig(Config):
    """ Hyperparameters for retraining

    Derived from https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/config.py
    """
    # give the configuration a recognizable name
    NAME = "MaskRCNN_config"
 
    # set the number of GPUs to use along with the number of images
    # per GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
 
    # number of classes (we would normally add +1 for the background)
    # concrete barrier + metal barrier + rumble strip + BG
    NUM_CLASSES = 3+1
   
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 131
    
    # Learning rate
    LEARNING_RATE=0.006
    
    # Skip detections with < 70% confidence
    DETECTION_MIN_CONFIDENCE = 0.7
    
    # setting Max ground truth instances
    MAX_GT_INSTANCES=100