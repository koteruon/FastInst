#!/bin/bash
read -p "Enter image source folder: " image_folder
image_folder="""+image_folder+"""
python chinese_detect.py --weights 0926_best.pt --conf 0.35 --img-size 640 --source image_folder --save-txt --save-conf --no-trace