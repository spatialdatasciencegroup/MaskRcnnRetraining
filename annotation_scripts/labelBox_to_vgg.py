import sys, getopt
import json
import cv2
import numpy as np

from typing import List

def main(argv):
    """ Creates the appropriate annotations from segmentation annotations
    created in Labelbox.

    The annotations should be in the format: 
    {
        "image_name.jpg": {
            "image_id": 0,
            "width": 640,
            "height": 640,
            "annotations": [
                {
                    "category": "concretebarrier",
                    "annotation": [
                {
                    "x": 0,
                    "y": 300
                },
                {
                    "x": 2,
                    "y": 500
                },
                ...
            ]
        },
        "image_name_2.jpg: {...},
        ...
    }

    If you create your segmentations using some other tool, a new conversion
    script may have to be created to create annotations in the above format.
    """
    # default files
    input_file = './road-blocks.json'
    output_file = './road-blocks-custom.json'
    # handle arguments and options
    try:
        opts, args = getopt.getopt(argv, 'i:o:', ['ifile', 'ofile'])
    except getopt.GetoptError:
        print('Incorrect usage.')
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-i', '--ifile'):
            input_file = arg
        elif opt in ('-o', '--ofile'):
            output_file = arg

    # massage labelBox json data into VGG format...
    with open(input_file, 'r') as labelBox_file:
        input_arr = json.load(labelBox_file)

    images_dict = construct_images_dict(input_arr)

    with open(output_file, 'w') as vgg_file:
        vgg_file.write(json.dumps(images_dict))

def construct_images_dict(input_arr: List) -> dict:
    """ Constructs a dictionary of images where the key is the filename
    of the image and the values are the image_id of the image and
    its corresponding annotations

    Also stores a list of unique categories under the "categories" key
    """
    images_dict = {}
    unique_categories = set()
    image_id = 0
    for image_dict in input_arr:
        # get image name
        image_name = image_dict['External ID']
        print(image_name)
        # get image shape
        height, width, _ = get_image_shape(image_name)

        # get annotations associated with image
        annotations = []
        for labelbox_annotation in image_dict['Label']['objects']:
            category = labelbox_annotation['value']
            polygon = labelbox_annotation['polygon']

            # TODO store width and height for each image
            annotations.append({
                'category': category,
                'annotation': polygon
            })

            unique_categories.add(category)
        
        images_dict[image_name] = {}
        images_dict[image_name]['image_id']    = image_id
        images_dict[image_name]['width']       = width
        images_dict[image_name]['height']      = height
        images_dict[image_name]['annotations'] = annotations

        image_id += 1

    images_dict['categories'] = list(unique_categories)

    return images_dict

def get_image_shape(image_name: str) -> any:
    """ Gets an image's shape using openCV

    If the image is not found in the train directory, will look
    in the validation directory
    """
    base_path = './road-block-images/'

    image = cv2.imread(base_path + 'train/' + image_name)
    if type(image) == np.ndarray:
        pass
    else:
        print(image_name + ' not in /train.')
        image = cv2.imread(base_path + 'val/' + image_name)

    print(type(image))
    print(image)
    return image.shape

if __name__ == '__main__':
    main(sys.argv[1:])