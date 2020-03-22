from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)

    # Get data configuration.
    data_config = parse_data_config(opt.data_config)
    test_path = data_config["test"]
    classes = data_config["names"]

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model = torch.load(opt.weights_path, map_location=device)

    model.eval()  # Set in evaluation mode

    # Get dataloader
    dataset = ListDataset(test_path, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index
    target_boxes = [] # Stores target bounding boxes.

    print("\nPerforming object detection:")
    prev_time = time.time()
    for batch_i, (img_paths, input_imgs, targets) in enumerate(dataloader):
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

        print(f"\t\tPaths: {img_paths}")
        print(f"\t\tDetections: {detections}")
        print(f"\t\tTargets: {targets}")

        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)
        target_boxes.append(targets)

    # Bounding-box colors
    cmap = plt.get_cmap("bone")

    # This project only has two of them.
    colors = ["#fc9272", "#addd8e"]

    # Create plot.
    plt.figure()

    print("\nSaving images:")
    # Iterate through images and save plot of detections
    for img_i, (path, detections, targets) in enumerate(zip(imgs,
                                                            img_detections,
                                                            target_boxes)):
        print("(%d) Image: '%s'" % (img_i, path))

        # Create plot.
        img = np.array(Image.open(path))
        fig, ax = plt.subplots(1)
        ax.imshow(img, cmap=cmap)

        if targets is not None:
            targets = targets[:, 2:6]
            targets = xywh2xyxy(targets * opt.img_size)

            for index, (x1, y1, x2, y2) in enumerate(targets):
                print("\t- Label: Target, Box: %s" % str(targets))

                box_w = x2 - x1
                box_h = y2 - y1

                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), box_w, box_h,
                                         linewidth=2, edgecolor=colors[1],
                                         facecolor="none")
                # Add the bbox to the plot
                ax.add_patch(bbox)

        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

                box_w = x2 - x1
                box_h = y2 - y1

                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), box_w, box_h,
                                         linewidth=2, edgecolor=colors[0],
                                         facecolor="none")
                # Add the bbox to the plot
                ax.add_patch(bbox)
                # Add label
                plt.text(
                    x1,
                    y1,
                    s="%.0f%%" % (cls_conf.item() * 100),
                    color="white",
                    verticalalignment="top",
                    bbox={"color": colors[0], "pad": 0},
                )

        # Save generated image with detections
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        filename = path.split("/")[-1].split(".")[0]
        plt.savefig(f"output/{filename}.png", bbox_inches="tight", pad_inches=0.0)
        plt.clf()
