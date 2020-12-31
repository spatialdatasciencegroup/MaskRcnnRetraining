# An example images/ directory structure

The images directory should contain both a train and a validation set of images each with their own corresponding annotations stored as a json file. For an example of the annotations file, see [val/annotations.json](./val/annotations.json).

The directory structure should be as follows:
```
images/
    train/
        images/
            img1.jpg
            img2.jpg
            ...
        annotations.json
    val/
        images/
            img1.jpg
            img2.jpg
            ...
        annotations.json
```

The annotation format for each dataset (train and validation) should be as follows:

```
{
    "image_name.jpg": {
        "image_id": 0,
        "width": 640,
        "height": 640,
        "annotations": [
            {
                "category": "dog",
                "annotation": [
                    { "x": 0, "y": 383.565 },
                    { "x": 322.173, "y": 331.952 },
                    ...
                ]
            },
            {
                "category": "cat",
                "annotation": [
                    ...
                ]
            }
        ]
    },
    "image_name_2.jpg": {
        "image_id": 1,
        ...
    },
    ...
    "categories": [
        "dog",
        "cat",
        "platypus",
        ...
    ]
}
```

where the keys/values are described as follows:

* **image_id**: the image_id is an arbitrary unique, integer id for each image.
* **width and height**: are the pixel dimensions of the image
* **annotations**: a list of annotated objects contained within the image
    * arbitrary length
    * can contain different categories within the same image
    * each annotation consists of...
        1. The class-category of the object
        2. a list of (x,y) pairs of arbitrary length that describes where the object is in the image (a polygon)
* **categories**: a list of the available categories in the dataset