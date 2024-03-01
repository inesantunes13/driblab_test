"""
    Prepare data from YOLO inference and re-do classification using k-means
"""

# Imports
import os
import argparse
import cv2
import re
import sys
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# Function to calculate color histogram feature vector
def calculate_color_histogram(image):
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = hist.flatten()
    return hist


def extract_numeric_value(filename):
    return [int(x) if x.isdigit() else x for x in re.split(r'(\d+)', filename)]


def prepare_frame_info(file_directory: str)-> pd.DataFrame:
    """Receives file path to txt files with information about each frame and the objects detected in them.
    Organizes all the information in one single dataframe with information per frame and all the objects detected (id: unique identifier of all objects,
    cls: class given by pre-trained yolo, xyxy and xywh)
    Args:
        file_directory (str): _description_

    Returns:
        pd.DataFrame: all frame info organzed and prepared for further use
    """    
    
    # Get a sorted list of .txt files in the directory using the custom sorting function
    txt_files = sorted([f for f in os.listdir(file_directory) if f.endswith(".txt")], key=extract_numeric_value)

    # Initialize a list to store all frames information
    all_frames_info = pd.DataFrame(index=range(len(txt_files)),columns=["clip_name", "frame", "objects_info"])

    for frame_nr, txt_file in enumerate(txt_files):
        file_path = os.path.join(file_directory, txt_file)
        
        # Read the content of the .txt file
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Parse the information from each line of the .txt file
        objects_info = pd.DataFrame(columns = ["id", "cls", "xyxy", "xywh"])
        xywh_column = []
        xyxy_column = []
        id_column = []
        cls_column = []
        
        for obj_nr, line in enumerate(lines):
            # Assuming each line has the format: "class x_center y_center width height"
            cls, x_t, y_t, x_tt, y_tt, x_center, y_center, width, height = line.strip().split()

            # Extract frame number and clip name from the file name
            _, _, frame_number = txt_file.split('_')
            frame_number = frame_number.split('.')[0]
            id = '_'.join([frame_number, str(obj_nr)])
            xywh = [float(x_center),float(y_center),float(width),float(height)]
            xyxy = [int(x_t),int(y_t),int(x_tt),int(y_tt)]
            id_column.append(id)
            cls_column.append(cls)
            xywh_column.append(xywh)
            xyxy_column.append(xyxy)
            
            
        objects_info["id"] = id_column
        objects_info["cls"] = cls_column
        objects_info["xywh"] = xywh_column
        objects_info["xyxy"] = xyxy_column
        
        # frame_infos[obj_nr] = pd.DataFrame(data = {"frame":})
        frame = txt_file.split('_')[-1].split('.')[0]
        clip_name = txt_file.rsplit('_', 1)[0]
        all_frames_info.iloc[frame_nr] = [clip_name, frame, objects_info]
        
    return all_frames_info


def calculate_objects_features(df_frames_info:pd.DataFrame, crops_folder_path: str) -> pd.DataFrame:
    """It receives a dataframe with all detected objects, filters by detection and uses the crops to calculate color histogram feature.
    It groups all the features in a single dataframe.
    Args:
        df_frames_info (pd.DataFrame): dataframe with all information from all the objects detected per frame
        crops_folder_path (str): path to the folder with all crops saved
    Returns:
        pd.DataFrame: dataframe with object unique identifier and its color histogram feature
    """    

    # Initialize DataFrame to save features
    obj_features = pd.DataFrame(columns = ["id", "color_hist"])
    
    # Process each crop
    for _, frame_info in df_frames_info.iterrows():  # Loop through each frame's information
        
        objects = frame_info["objects_info"]
        
        # only will analyse images classified as "person"
        object_players = objects[objects["cls"] != '32'].reset_index(drop=True) 
        
        color_hist = [None]* len(object_players)
        id_objects = [None]* len(object_players)
        for obj_nr,  obj_info in object_players.iterrows():  # Loop through each object in the frame
            
            crop_path = f"{crops_folder_path}/{frame_info['clip_name']}_{obj_info['id']}.jpg"
            
            # Read the crop image
            crop_image = cv2.imread(crop_path)
            
            # Calculate color histogram feature vector
            feature_vector = calculate_color_histogram(crop_image)
            
            color_hist[obj_nr] = feature_vector
            
            id_objects[obj_nr] = str(obj_info['id'])
            
            frame_obj_features = {"id" : id_objects, "color_hist": color_hist}
            
            frame_obj_features = pd.DataFrame(frame_obj_features)
                
        # Add information to DataFrame
        obj_features = pd.concat([obj_features, frame_obj_features], axis = 0)
        
    obj_features.reset_index(drop=True, inplace=True)
    
    return obj_features
            

def classify_using_kmeans(folder_path:str) -> pd.DataFrame:
    """It receives the path for model results with all the object detection information per frame, processes and prepares that information, calculates
    relevant feature for all objects and uses k-means to classify objects initially classified as "person":

    Args:
        folder_path (str): path to the folder with detection information per frame.
    """    
    txt_folder_path = folder_path + "/labels/"
    crops_folder_path = folder_path + "/crops/"
    
    # Read and organize detection information for each frame of the video received
    frames_info = prepare_frame_info(txt_folder_path)
    
    # calculate features for kmeans:
    obj_features = calculate_objects_features(frames_info, crops_folder_path)

    # feature normalization
    scaler = StandardScaler()
    normalized_hist_standard = scaler.fit_transform(np.array(obj_features['color_hist'].tolist()))

    # Perform k-means clustering with 3 clusters
    kmeans = KMeans(n_clusters=3, random_state=42)
    obj_features['cluster'] = kmeans.fit_predict(normalized_hist_standard)
    
    # update class of objetcs after k-means classification
    for frame_index, frame_info in frames_info.iterrows():
        objects_info = frame_info['objects_info']
        obj_class_column = [None] * len(objects_info)
        
        for obj_ind, obj in objects_info.iterrows():
            if obj['id'] in obj_features['id'].values:
                obj_class = obj_features[obj_features['id'] == obj['id']]['cluster'].iloc[0]
            else:
                obj_class = int(obj['cls'])
            obj_class_column[obj_ind] = obj_class
            
        frames_info.iloc[frame_index]["objects_info"]["cls"] = obj_class_column
            
    return frames_info


# if __name__ == "__main__":
#     # Replace 'your_folder_path' with the actual path to the folder containing detection information per frame
#     folder_path = 'your_folder_path'
    
#     # Call the function for debugging
#     frames_info = classify_using_kmeans(folder_path)


    











    
    
    

