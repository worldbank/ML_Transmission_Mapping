import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.applications.resnet import ResNet50, ResNet101, ResNet152
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    Dropout,
    Lambda,
    ReLU,
)
from tensorflow.keras.regularizers import l2

from model.centernet_utils import get_affine_transform


def nms(heat, kernel=3):
    """
    Non maximal supression for a heatmap

    Args:
        heat: Heatmap (class predictions of centernet)
        kernel: The kernel size to apply the nms with

    Returns:
        The heatmap, with everything that is non-maximal supressed ;)
    """
    hmax = tf.nn.max_pool2d(heat, (kernel, kernel), strides=1, padding="SAME")
    heat = tf.where(tf.equal(hmax, heat), heat, tf.zeros_like(heat))
    return heat


def topk(hm, max_objects=100):
    """
    Select the top k points in the heatmap with the highest value, these are then evaluated if they're objects
    (if their score is high enough)
    Args:
        hm: The heatmap
        max_objects: Number of objects to inspec

    Returns:
        scores, indices, class_ids, xs, ys
        i.e. the information about these max_objects points on the heatmap
    """
    hm = nms(hm)
    # (b, h * w * c)
    b, _, w, c = tf.shape(hm)[0], tf.shape(hm)[1], tf.shape(hm)[2], tf.shape(hm)[3]
    hm = tf.reshape(hm, (b, -1))
    # (b, k), (b, k)
    scores, indices = tf.nn.top_k(hm, k=max_objects)
    class_ids = indices % c
    xs = indices // c % w
    ys = indices // c // w
    indices = ys * w + xs
    return scores, indices, class_ids, xs, ys


def evaluate_batch_item_boxes(
    batch_item_detections,
    num_classes,
    max_objects_per_class=20,
    max_objects=100,
    iou_threshold=0.5,
    score_threshold=0.1,
):
    """
    Filters the batch_item_boxes on score_threshold, and the nms and iou
    Args:
        batch_item_detections: the detections
        num_classes: number of classes to expect
        max_objects_per_class: the max number of objects per class to predict
        max_objects: the maximum number of total objects to consider
        iou_threshold: the maximum overlap between detections before filtering one of them out
        score_threshold: the minimum score to be considered a proper detection

    Returns:
        The filtered detections
    """
    batch_item_detections = tf.boolean_mask(
        batch_item_detections, tf.greater(batch_item_detections[:, 4], score_threshold)
    )
    detections_per_class = []
    for cls_id in range(num_classes):
        class_detections = tf.boolean_mask(
            batch_item_detections, tf.equal(batch_item_detections[:, 5], cls_id)
        )
        nms_keep_indices = tf.image.non_max_suppression(
            class_detections[:, :4],
            class_detections[:, 4],
            max_objects_per_class,
            iou_threshold=iou_threshold,
        )
        class_detections = tf.gather(class_detections, nms_keep_indices)
        detections_per_class.append(class_detections)

    batch_item_detections = tf.concat(detections_per_class, axis=0)

    def filter():
        nonlocal batch_item_detections
        _, indices = tf.nn.top_k(batch_item_detections[:, 4], k=max_objects)
        batch_item_detections_ = tf.gather(batch_item_detections, indices)
        return batch_item_detections_

    def pad():
        nonlocal batch_item_detections
        batch_item_num_detections = tf.shape(batch_item_detections)[0]
        batch_item_num_pad = tf.maximum(max_objects - batch_item_num_detections, 0)
        batch_item_detections_ = tf.pad(
            tensor=batch_item_detections,
            paddings=[[0, batch_item_num_pad], [0, 0]],
            mode="CONSTANT",
            constant_values=0.0,
        )
        return batch_item_detections_

    batch_item_detections = tf.cond(tf.shape(batch_item_detections)[0] >= max_objects, filter, pad)
    return batch_item_detections


def decode_points(
    hm,
    reg,
    max_objects=100,
    nms=True,
    flip_test=False,
    num_classes=20,
    score_threshold=0.1,
    overlap_buffer=50,
):
    """
    The outputs from centernet are fairly abstract, this function takes the outputs from the various "heads" and
    decodes them into point predictions
    Args:
        hm: The heatmap outputted by the network, denotes classes and centerpoints
        reg: The regression offset, since the prediction is downsampled 4x, this is needed for more accuracy
        max_objects: The max number of objects to predict
        nms: Whether to apply NMS
        flip_test: When true, pass the image twice, once flipped, and combine the results
        num_classes: The number of classes to predict
        score_threshold: The minimum score to be considered as a detection
        overlap_buffer: The size of the area around the point to check overlap with

    Returns:
        an array of 100,4 (x, y, score, class)
    """
    if flip_test:
        hm = (hm[0:1] + hm[1:2, :, ::-1]) / 2
        reg = reg[0:1]
    scores, indices, class_ids, xs, ys = topk(hm, max_objects=max_objects)
    b = tf.shape(hm)[0]
    # (b, h * w, 2)
    reg = tf.reshape(reg, (b, -1, tf.shape(reg)[-1]))
    # (b, h * w, 2)
    topk_reg = tf.gather(reg, indices, batch_dims=1)
    # (b, k, 2)
    topk_cx = (tf.cast(tf.expand_dims(xs, axis=-1), tf.float32) + topk_reg[..., 0:1]) * 4
    topk_cy = (tf.cast(tf.expand_dims(ys, axis=-1), tf.float32) + topk_reg[..., 1:2]) * 4
    topk_lx = topk_cx - overlap_buffer // 2
    topk_ly = topk_cy - overlap_buffer // 2
    topk_rx = topk_cx + overlap_buffer // 2
    topk_ry = topk_cy + overlap_buffer // 2
    scores = tf.expand_dims(scores, axis=-1)
    class_ids = tf.cast(tf.expand_dims(class_ids, axis=-1), tf.float32)
    # (b, k, 6)
    detections = tf.concat(
        [topk_lx, topk_ly, topk_rx, topk_ry, scores, class_ids, topk_cx, topk_cy], axis=-1
    )
    if nms:
        detections = tf.map_fn(
            lambda x: evaluate_batch_item_boxes(
                x, num_classes=num_classes, score_threshold=score_threshold
            ),
            elems=detections,
        )
    detections = tf.concat(
        [detections[..., 6:7], detections[..., 7:8], detections[..., 4:5], detections[..., 5:6]],
        axis=-1,
    )
    return detections


def centernet(
    num_classes,
    backbone="resnet50",
    max_objects=100,
    score_threshold=0.1,
    nms=True,
    flip_test=False,
):
    """
    The actual model definition
    It is basically just some "heads" stacked on a generic backbone, each head learning a property of the prediction
    Args:
        num_classes: The number of classes to predict
        backbone: What backbone to use, currently only supports Resnets
        max_objects: The maximum number of objects to consider
        score_threshold: The minumum score before a detection is considered
        nms: Whether to apply nms
        flip_test: If True, the image is passed twice, once flipped, and the results are merged

    Returns:
    2 versions of the model:
    - prediction model (outputs points)
    - debug model (outputs the raw output of the heads)
    """
    backbones = {
        "resnet50": ResNet50,
        "resnet101": ResNet101,
        "resnet152": ResNet152,
    }
    assert backbone in backbones.keys()

    image_input = Input(shape=(None, None, 3), name="batch_images")

    resnet = backbones[backbone](
        input_shape=(None, None, 3), input_tensor=image_input, include_top=False
    )

    # (b, 16, 16, 2048)
    C5 = resnet.outputs[-1]
    # C5 = resnet.get_layer('activation_49').output

    x = Dropout(rate=0.5)(C5)
    # decoder
    num_filters = 256
    for i in range(3):
        num_filters = num_filters // pow(2, i)
        x = Conv2DTranspose(
            num_filters,
            (4, 4),
            strides=2,
            use_bias=False,
            padding="same",
            kernel_initializer="he_normal",
            kernel_regularizer=l2(5e-4),
        )(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

    # hm header
    y1 = Conv2D(
        64,
        3,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=l2(5e-4),
    )(x)
    y1 = BatchNormalization()(y1)
    y1 = ReLU()(y1)
    y1 = Conv2D(
        num_classes,
        1,
        kernel_initializer="he_normal",
        kernel_regularizer=l2(5e-4),
        activation="sigmoid",
        name="hm_pred",
    )(y1)

    # reg header
    y3 = Conv2D(
        64,
        3,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=l2(5e-4),
    )(x)
    y3 = BatchNormalization()(y3)
    y3 = ReLU()(y3)
    y3 = Conv2D(2, 1, kernel_initializer="he_normal", kernel_regularizer=l2(5e-4), name="reg_pred")(
        y3
    )

    # detections = decode(y1, y3)
    detections = Lambda(
        lambda x: decode_points(
            *x,
            max_objects=max_objects,
            score_threshold=score_threshold,
            nms=nms,
            flip_test=flip_test,
            num_classes=num_classes
        )
    )([y1, y3])

    prediction_model = Model(inputs=image_input, outputs=detections)

    debug_model = Model(inputs=image_input, outputs=[y1, y3])

    return prediction_model, debug_model
