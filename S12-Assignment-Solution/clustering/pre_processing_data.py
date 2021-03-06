"""Preprocess and cleanup data."""
import pandas as pd
import numpy as np
import json
import cv2
import os


def normalise_data(row):
  """."""
  try:
      region_shape_dict = json.loads(row["region_shape_attributes"])
      region_attributes_dict = json.loads(row["region_attributes"])
  except:
      return row

  row["CX"] = region_shape_dict["x"]
  row["CY"] = region_shape_dict["y"]
  row["H"] = region_shape_dict["height"]
  row["W"] = region_shape_dict["width"]
  row["class_label"] = region_attributes_dict["name"]
  return row


def normalise_annotations_dataframe(df):
    """."""
    df["CX"] = np.nan
    df["CY"] = np.nan
    df["H"] = np.nan
    df["W"] = np.nan
    df = df.apply(lambda row: normalise_data(row), axis=1)
    df = df.loc[~df["CX"].isnull()].copy()

    bbox_df = pd.DataFrame()
    bbox_df['CX'] = df["CX"].copy()
    bbox_df['CY'] = df["CY"].copy()
    bbox_df['H'] = df["H"].copy()
    bbox_df['W'] = df["W"].copy()
    bbox_df['class_label'] = df["class_label"].copy()
    bbox_df["filename"] = df["filename"].copy()

    return bbox_df


def get_image_size_info(images_path):
    """."""
    image_files = [
        x for x in os.listdir(images_path)
        if x.endswith(".jpg")
    ]
    image_height_list = []
    image_width_list = []
    for image_file in image_files:
        image = cv2.imread(os.path.join(images_path, image_file ) )
        height, width, channels = image.shape
        image_height_list.append(height)
        image_width_list.append(width)

    images_df = pd.DataFrame()
    images_df["image_file_name"] = image_files
    images_df["img_h"] = image_height_list
    images_df["img_w"] = image_width_list

    return images_df