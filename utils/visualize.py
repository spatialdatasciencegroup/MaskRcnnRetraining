from mrcnn import visualize
from mrcnn import model as modellib

from MaskRCNNConfig import MaskRCNNConfig
from RoadBarrierDataset import RoadBarrierDataset

def visualize_dataset(dataset: RoadBarrierDataset, config: MaskRCNNConfig, 
        model: any = None, limit: int = 0) -> None:
    """ Visualizes the given dataset

    Visualizes the ground truth segmentations by default. However, if
    a model is passed it runs detections on the images in the dataset
    and displays those detections instead.

    An optional limit parameter can also be passed that specifies
    the limit of the number of images you want displayed.
    """
    for index, image_id in enumerate(dataset.image_ids):
        image, image_meta, gt_class_id, gt_bbox, gt_mask =    modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
        info = dataset.image_info[image_id]
        print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, 
                                                dataset.image_reference(image_id)))

        if not model: # use ground truth values
            perfect_scores = [1] * len(gt_class_id) # provide dummy scores of 100% accuracy

            visualize.display_instances(image, gt_bbox, gt_mask, gt_class_id, 
                                    dataset.class_names, perfect_scores, 
                                    title="Ground Truth")
        else:
            # Run object detection
            results = model.detect([image], verbose=1)
            # Display results

            r = results[0]
            visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                        dataset.class_names, r['scores'], 
                                        title="Predictions")
        
        # check limit
        if limit > 0 and index == limit - 1:
            return
            
