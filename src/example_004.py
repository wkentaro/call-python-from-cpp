#!/usr/bin/env python

import chainer
from chainercv.datasets import coco_instance_segmentation_label_names
from chainercv.links import MaskRCNNFPNResNet50
import imgviz
import numpy as np


def main():
    gpu = 0
    image_file = '../data/33823288584_1d21cf0a26_k.jpg'

    chainer.config.cv_resize_backend = 'cv2'

    label_names = coco_instance_segmentation_label_names
    model = MaskRCNNFPNResNet50(
        n_fg_class=len(label_names),
        pretrained_model='coco',
    )

    chainer.cuda.get_device_from_id(gpu).use()
    model.to_gpu()

    img = imgviz.io.imread(image_file)
    img_input = img.transpose(2, 0, 1).astype(np.float32)

    masks, labels, scores = model.predict([img_input])
    masks = masks[0]
    labels = labels[0]
    scores = scores[0]
    class_ids = labels + 1
    captions = [
        '{:s}: {:.2%}'.format(label_names[c - 1], s)
        for c, s in zip(class_ids, scores)
    ]
    viz = imgviz.instances2rgb(
        image=img,
        labels=class_ids,
        masks=masks,
        captions=captions,
        font_size=15,
    )

    imgviz.io.pyglet_imshow(viz)
    imgviz.io.pyglet_run()


if __name__ == '__main__':
    main()
