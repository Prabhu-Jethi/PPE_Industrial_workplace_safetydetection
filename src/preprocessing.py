import os
import yaml
import pandas as pd
import cv2

DATA_DIR = '../data'
YAML_FILE = os.path.join(DATA_DIR, 'data.yaml')
TRAIN_IMAGES = os.path.join(DATA_DIR, 'train', 'images')
TRAIN_LABELS = os.path.join(DATA_DIR, 'train', 'labels')


