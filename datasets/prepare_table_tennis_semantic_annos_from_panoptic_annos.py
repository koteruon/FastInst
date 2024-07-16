#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import functools
import json
import multiprocessing as mp
import os
import time

import cv2
import numpy as np
from detectron2.data.datasets.builtin_meta import TABLE_TENNIS_CATEGORIES
from panopticapi.utils import rgb2id
from PIL import Image, ImageDraw


def _process_panoptic_to_semantic(input, output_panoptic, output_semantic, segments, id_map):
    panoptic = np.asarray(Image.open(input), dtype=np.uint32)
    panoptic = Image.fromarray(np.zeros_like(panoptic, dtype=np.uint8))
    semantic = Image.fromarray(np.zeros_like(panoptic, dtype=np.uint8) + 255)

    # panoptic
    panoptic_output = ImageDraw.Draw(panoptic)
    for seg in segments:
        cat_id = seg["category_id"]
        color = id_map[cat_id]["color"]
        for id in seg["id"]:
            polygon = [(id[i], id[i + 1]) for i in range(0, len(id), 2)]
            panoptic_output.polygon(polygon, fill=tuple(color))
    panoptic.save(output_panoptic)

    # semantic
    semantic_output = ImageDraw.Draw(semantic)
    for seg in segments:
        cat_id = seg["category_id"]
        new_cat_id = id_map[cat_id]["semantic_id"]
        for id in seg["id"]:
            polygon = [(id[i], id[i + 1]) for i in range(0, len(id), 2)]
            semantic_output.polygon(polygon, fill=new_cat_id)
    semantic.save(output_semantic)


def separate_coco_semantic_from_panoptic(panoptic_json, image_root, panoptic_root, sem_seg_root, categories):
    """
    Create semantic segmentation annotations from panoptic segmentation
    annotations, to be used by PanopticFPN.
    It maps all thing categories to class 0, and maps all unlabeled pixels to class 255.
    It maps all stuff categories to contiguous ids starting from 1.
    Args:
        panoptic_json (str): path to the panoptic json file, in COCO's format.
        panoptic_root (str): a directory with panoptic annotation files, in COCO's format.
        sem_seg_root (str): a directory to output semantic annotation files
        categories (list[dict]): category metadata. Each dict needs to have:
            "id": corresponds to the "category_id" in the json annotations
            "isthing": 0 or 1
    """
    os.makedirs(panoptic_root, exist_ok=True)
    os.makedirs(sem_seg_root, exist_ok=True)

    id_map = {}  # map from category id to id in the output semantic annotation
    assert len(categories) <= 254
    for i, k in enumerate(categories):
        id_map[k["id"]] = {"color": k["color"], "semantic_id": k["id"]}
    # what is id = 0?
    # id_map[0] = 255
    print(id_map)

    with open(panoptic_json) as f:
        obj = json.load(f)

    pool = mp.Pool(processes=max(mp.cpu_count() // 2, 4))

    def iter_annotations():
        image_file_index = {}
        for image in obj["images"]:
            image_file_index[image["id"]] = os.path.basename(image["path"])
        image_seg_index = {}
        for anno in obj["annotations"]:
            if anno["image_id"] in image_seg_index:
                image_seg_index[anno["image_id"]].append(
                    {"category_id": anno["category_id"], "id": anno["segmentation"]}
                )
            else:
                image_seg_index[anno["image_id"]] = [{"category_id": anno["category_id"], "id": anno["segmentation"]}]
        for anno in obj["annotations"]:
            file_name = image_file_index[anno["image_id"]]
            segments = image_seg_index[anno["image_id"]]
            input = os.path.join(image_root, file_name)
            panoptic_input = os.path.join(panoptic_root, file_name)
            sem_seg_output = os.path.join(sem_seg_root, file_name)
            yield input, panoptic_input, sem_seg_output, segments

    print("Start writing to {} ...".format(sem_seg_root))
    start = time.time()
    pool.starmap(
        functools.partial(_process_panoptic_to_semantic, id_map=id_map),
        iter_annotations(),
        chunksize=100,
    )
    print("Finished. time: {:.2f}s".format(time.time() - start))


if __name__ == "__main__":
    dataset_dir = os.path.join(os.getenv("DETECTRON2_DATASETS", "datasets"), "table_tennis")
    for s in ["val", "train"]:
        separate_coco_semantic_from_panoptic(
            os.path.join(dataset_dir, "annotations/panoptic_{}.json".format(s)),
            os.path.join(dataset_dir, "{}".format(s)),
            os.path.join(dataset_dir, "panoptic_{}".format(s)),
            os.path.join(dataset_dir, "panoptic_semseg_{}".format(s)),
            TABLE_TENNIS_CATEGORIES,
        )
