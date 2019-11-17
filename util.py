import tensorflow as tf
import glob
from PIL import Image
import numpy as np
import json


def make_dataset_for_directory(directory, batch_size, augment=True, shuffle=True, take_subset=-1):
    dogs_train = sorted(glob.glob('%s/dogs/*.jpg'%directory))
    cats_train = sorted(glob.glob('%s/cats/*.jpg'%directory))

    filenames = dogs_train + cats_train
    labels = [[1,0]] * len(dogs_train) + [[0,1]] * len(cats_train)    
    if take_subset < 0:
        take_subset = len(labels)
        
    print("Found %d images in '%s'"%(len(filenames), directory))

    # Reads an image from a file, decodes it into a dense tensor, and resizes it
    # to a fixed shape, range [0,1]
    def decode_image_file(filename, y):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string)
        return tf.cast(image_decoded, tf.float32) / 255.0, y
    
    def resize_image(x, y):
        return tf.image.resize_image_with_pad(x, 150, 150), y

    def color_aug(x: tf.Tensor) -> tf.Tensor:
        x = tf.image.random_hue(x, 0.08)
        x = tf.image.random_saturation(x, 0.6, 1.6)
        x = tf.image.random_brightness(x, 0.05)
        x = tf.image.random_contrast(x, 0.7, 1.3)
        return x
    
    def flip_aug(x: tf.Tensor) -> tf.Tensor:
        x = tf.image.random_flip_left_right(x)
        return x
    
    def zoom_aug(x: tf.Tensor) -> tf.Tensor:
        shape = tf.cast(tf.shape(x),tf.float32)
        return tf.image.random_crop(x, [shape[0]*0.8, shape[1]*0.8, 3])
    
    dataset = tf.data.Dataset.from_tensor_slices((tf.constant(filenames[:take_subset]), tf.constant(labels[:take_subset])))
    dataset = dataset.map(decode_image_file, num_parallel_calls=4)
    
    AUGMENT_REPEATS = 4
    
    if augment:
        print("Augmenting. Total number of samples: %d"%(len(labels) * AUGMENT_REPEATS))
        
        dataset = dataset.repeat(AUGMENT_REPEATS) # add 4 repeats per image, for augmentations
        
        # Add augmentations
        augmentations = [flip_aug, color_aug, zoom_aug]

        # Add the augmentations to the dataset
        for f in augmentations:
            # Apply the augmentation, run 4 jobs in parallel.
            dataset = dataset.map(lambda x, y: (tf.cond(tf.random_uniform([], 0, 1) > 0.5, lambda: f(x), lambda: x), y), num_parallel_calls=4)

    dataset = dataset.map(resize_image, num_parallel_calls=4)
    
    # clip and normalize to [-1,1] range
    dataset = dataset.map(lambda x, y: (tf.clip_by_value(x, 0, 1) * 2.0 - 1.0, y), num_parallel_calls=4)

    # All the dogs are at the beginning and the cats in the end.... 
    # This way our model will first just learn to predict all 0s! We must shuffle the samples!
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(labels) * (AUGMENT_REPEATS if augment else 1), seed=42)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
        
    return dataset


def batch_iou(a, b, epsilon=1e-5):
    """ Given two arrays `a` and `b` where each row contains a bounding
        box defined as a list of four numbers: [x1,y1,x2,y2]
        Returns the Intersect of Union scores for each corresponding
        pair of boxes.
    """
    x1 = np.array([a[:, 0], b[:, 0]]).max(axis=0)
    y1 = np.array([a[:, 1], b[:, 1]]).max(axis=0)
    x2 = np.array([a[:, 2], b[:, 2]]).min(axis=0)
    y2 = np.array([a[:, 3], b[:, 3]]).min(axis=0)

    # AREAS OF OVERLAP - Area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)

    # handle case where there is NO overlap
    width[width < 0] = 0
    height[height < 0] = 0

    area_overlap = width * height

    # COMBINED AREAS
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    area_combined = area_a + area_b - area_overlap

    # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
    iou = area_overlap / (area_combined + epsilon)
    return iou


def make_detection_dataset_for_directory(directory, batch_size,bb_box_grid, shuffle=True, take_subset=-1):
    images = sorted(glob.glob('%s/*.jpg'%directory))
    jsons  = sorted(glob.glob('%s/*.json'%directory))
    if take_subset < 0:
        take_subset = len(jsons)
    print("Found %d images in '%s'"%(len(images), directory))
    
    labels = []
    for i,(json_file, image_file) in enumerate(zip(jsons,images)):
        with Image.open(image_file) as img:
            width, height = img.size
        scale = np.max([width,height]) / 150.
        pad_x = (150. - width / scale) / 2.
        pad_y = (150. - height / scale) / 2.

        y = [[0,0,0,0,0,0]] * len(bb_box_grid)
        with open(json_file, 'r') as f:
            for n,b in json.load(f):
                c = 1 if n == 'dog' else 0
                # transform BBox
                b_new = np.array(b) / scale + np.array([pad_x,pad_y,pad_x,pad_y])
                
                # match with BBox grid
                ious = batch_iou(np.int32(bb_box_grid), np.int32([b_new] * len(bb_box_grid)))

                # find grid-BBox[s] that has IoU > 0.25
                found_idxs = np.squeeze(np.argwhere(ious > 0.25))
                if np.count_nonzero(ious > 0.25) == 1: 
                    found_idxs = [found_idxs]

                for found_idx in found_idxs:
                    # Set [box residual/offset (normalized [-1,1]), class, objectness] in that grid-BBox
                    y[found_idx] = np.hstack([(bb_box_grid[found_idx] - b_new) / 150., c, 1])
                
        labels.append(np.float32(y).ravel())

    labels = np.array(labels)
            
    # Reads an image from a file, decodes it into a dense tensor, resizes and rescales to [-1,1]
    def decode_image_file(filename, y):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string)
        image_resized = tf.cast(tf.image.resize_image_with_pad(image_decoded, 150, 150), tf.float32) / 255.0
        # clip and normalize to [-1,1] range
        return tf.clip_by_value(image_resized, 0, 1) * 2.0 - 1.0, y

    
    dataset = tf.data.Dataset.from_tensor_slices((tf.constant(images[:take_subset]), 
                                                  tf.constant(labels[:take_subset])))
    dataset = dataset.map(decode_image_file, num_parallel_calls=4)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(jsons), seed=42)
        
    # Split training and validation: 90-10
    train_dataset = dataset.take(int(take_subset * 0.9))
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.repeat()
    
    val_dataset = dataset.skip(int(take_subset * 0.9))
    val_dataset = val_dataset.batch(1)
    val_dataset = val_dataset.repeat()
        
    return train_dataset, val_dataset


def show_boxes_offsets(im, y_boxes):
    ax = plt.gca()
    ax.imshow(im)
    for x in np.linspace(0,im.shape[1],5):
        ax.axvline(x)
    for y in np.linspace(0,im.shape[0],5):
        ax.axhline(y)
    for i,b in enumerate(y_boxes.reshape(len(bb_box_grid),6)):
        if b[-1] < 0.4:
            continue
        pred_bb = bb_box_grid[i] - (b[:4] * 150)
        
        rect = patches.Rectangle((pred_bb[0],pred_bb[1]),pred_bb[2]-pred_bb[0],pred_bb[3]-pred_bb[1],
                                 linewidth=1,edgecolor='r',facecolor='none',alpha=b[-1])
        ax.add_patch(rect)
        
        ax.text(pred_bb[0],pred_bb[1], 'cat' if b[-2] < 0.5 else 'dog', color='r', fontsize=20, alpha=b[-1])
    ax.set_xlim([0,150]),ax.set_ylim([150,0]);

def get_grid():
    bb_points_grid = np.dstack(np.meshgrid(np.linspace(0,150,5), np.linspace(0,150,5)))
    bb_box_grid = []
    for i in range(3):
        for j in range(3):
            bb_box_grid.append(np.hstack([bb_points_grid[i,j],bb_points_grid[i+2,j+2]]))
    for i in range(2):
        for j in range(2):
            bb_box_grid.append(np.hstack([bb_points_grid[i,j],bb_points_grid[i+3,j+3]]))
    bb_box_grid.append(np.hstack([bb_points_grid[0,0],bb_points_grid[-1,-1]]))
    return bb_box_grid