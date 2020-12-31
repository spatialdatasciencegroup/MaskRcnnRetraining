import skimage.draw
import json
import numpy as np
from os import listdir

from mrcnn.utils import Dataset

class RoadBarrierDataset(Dataset):
    """ This derived class loads our custom dataset into the format that
    the maskRCNN implementation requires.

    The "load_dataset" and "load_mask" functions are overriden here to
    load our custom dataset.

    Derived from the Dataset class: https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/utils.py
    """
    # load the dataset definitions
    def load_dataset(self, dataset_dir, is_train=True):
        """ Loads the dataset from the specified directory

        Dataset should be in the format:

        /dataset_dir
            /images
                image_1.jpg
                image_2.jpg
                ...
            annotations.json

        where "annotations.json" is the annotations file that describes
        each of the images found in /dataset_dir/images.
        """
        
        # Add classes.
        # Background is class 0
        self.add_class("dataset", 1, "concretebarrier")
        self.add_class("dataset", 2, "metalbarrier")
        self.add_class("dataset", 3, "rumblestrip")
        
        # define mapping from class name to class id
        category_to_id = {
            'concretebarrier': 1,
            'metalbarrier': 2,
            'rumblestrip': 3
        }
        
        # define data locations for images and annotations
        images_dir = dataset_dir + '/images/'
        annotations_file_name = dataset_dir + '/annotations.json'

        images_info = {}
        try:
          with open(annotations_file_name) as annotations_file:
            images_info = json.load(annotations_file)
        except IOError as e:
          print('Could not open annotations file for ' + dataset_dir + 'dataset.')
          raise IOError(e)
        
        # Iterate through all files in the folder to 
        # add class, images, and annotations
        for filename in listdir(images_dir):

            # extract image information from annotations
            image_id    = images_info[filename]['image_id']
            width       = images_info[filename]['width']
            height      = images_info[filename]['height']
            annotations = images_info[filename]['annotations']
            
            # setting image file
            img_path = images_dir + filename
            
            # adding images and annotations to dataset
            self.add_image(
                source='dataset', 
                image_id=image_id, 
                path=img_path,
                width=width,
                height=height,
                annotations=annotations, # a list of segmentations
                category_to_id=category_to_id
            )

    # load the masks for an image
    """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
     """
    def load_mask(self, image_id):
        """ Uses the loaded dataset to construct a mask for the given
        image (identified by its image_id).
        """
        # get details of image
        info = self.image_info[image_id]

        # get information associated with image
        annotations    = info['annotations']
        width          = info['width']
        height         = info['height']
        category_to_id = info['category_to_id']

        # convert polygons to a bitmap mask of shape
        #   [height, width, instance_count]
        mask = np.zeros([height, width, len(annotations)],
                        dtype=np.uint8)
        class_ids = []

        for index, annotation_info in enumerate(annotations):
          category = annotation_info['category']
          annotation = annotation_info['annotation'] # a polygon

          category_id = category_to_id[category]

          all_points_x = [int(point['x']) for point in annotation]
          all_points_y = [int(point['y']) for point in annotation]

          # get indices of pixels inside the polygon and set them to 1
          rr, cc = skimage.draw.polygon(all_points_y, all_points_x)
          mask[rr, cc, index] = category_id

          # add category id to list of class ids
          class_ids.append(category_id)

        return mask, np.array(class_ids)

    # load an image reference
    """Return the path of the image."""
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        print(info)
        return info['path']