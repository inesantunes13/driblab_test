"""
Receives new classifications and outputs the intented files:
- re-annotated video with new classes 
- json file with information every 5 frames
- new labels/ and crops/ information saved

"""

import argparse
import os
import platform
import sys
from pathlib import Path
import pandas as pd
import json 

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages
from utils.general import (
    LOGGER,
    Profile,
    check_img_size,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    print_args,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()

def re_process_results(
    source, # video location
    new_classification_file, # file with new classifications
    imgsz=(640, 640),  # inference size (height, width)
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=False,  # save results to *.txt
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    project=ROOT / "runs/detect",  # save results to project/name
    name="exp",  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    classes_detect_info = pd.read_pickle(new_classification_file)
    
    #change type of some columns:
    classes_detect_info["frame"] = classes_detect_info["frame"].astype(int)
    save_img = not nosave and not source.endswith(".txt")  # save frames

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    (save_dir / "jsons").mkdir(parents=True, exist_ok=True)  # make dir

    # Define params for dataloader
    device = select_device(device)
    _, windows, _ = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
    class_names = {0: "team_1", 1: "team_2", 2: "referee", 32: "ball"}
    stride, names, pt = 32, class_names, True
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # initialize dictionary to save json file data
    all_json_data = {}
    
    # Iterate over frames of video and rows of df with location and class information 
    for path, _, im0s, vid_cap, s in dataset:
        p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)
        obj_info = classes_detect_info[classes_detect_info["frame"] == frame]["objects_info"].iloc[0]

        # get info every 5 frames
        if (frame-1) % 5 == 0:
            
            #prepare df with nr of objects per class
            nr_per_class = obj_info.groupby("cls").count().reset_index()
            nr_per_class = nr_per_class[["cls", "id"]]
            nr_per_class.rename(columns={"cls": "cls", "id": "nr_cls"}, inplace=True)
            nr_per_class["label"] = [class_names[c] for c in nr_per_class["cls"]]
            
            # get ball location in pixeis
            if "ball" in nr_per_class["label"].str.strip().values: #ball was detected
                ball_obj = obj_info[obj_info["cls"] == 32]

                ball_xyxy = [torch.tensor(float(val), device='cuda:0') for val in ball_obj["xyxy"].iloc[0]]
                ball_loc = xyxy2xywh(torch.tensor(ball_xyxy).view(1, 4)).view(-1).tolist() #xywh in pixels
                
            else:
                ball_loc = "ball not detected in this frame"
            
            nr_team_1 = nr_per_class[nr_per_class["label"] == "team_1"]["nr_cls"].iloc[0]
            nr_team_2 = nr_per_class[nr_per_class["label"] == "team_2"]["nr_cls"].iloc[0]
            
            if "referee" in nr_per_class["label"].str.strip().values: #referee was detected
                nr_referee = nr_per_class[nr_per_class["label"] == "referee"]["nr_cls"].iloc[0]
            else:
                nr_referee = 0
            
            json_5th_frame = {
                "frame": int(frame - 1),
                "team_1": int(nr_team_1),
                "team_2": int(nr_team_2), 
                "refs": int(nr_referee), 
                "ball_loc": ball_loc if isinstance(ball_loc, str) else tuple(int(val) for val in ball_loc)
            }

            # Store the current frame's data in the overarching dictionary
            all_json_data[f'json_{frame - 1}'] = json_5th_frame

        p = Path(p)
        save_path = str(save_dir / p.name) 
        txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        imc = im0.copy() if save_crop else im0  # for save_crop
        annotator = Annotator(im0, line_width=line_thickness, example=str(names))
        
        # Write results
        for _, obj in obj_info.iterrows():
            xyxy = [torch.tensor(float(val), device='cuda:0') for val in obj["xyxy"]]
            c = int(obj["cls"])  # integer class
            label = names[c]
            
            if save_txt:  # Write to file
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                line = (obj["cls"], *xywh) # label format
                with open(f"{txt_path}.txt", "a") as f:
                    f.write(("%g " * len(line)).rstrip() % line + "\n")

            if save_img or save_crop or view_img:  # Add bbox to image
                label = names[c]
                annotator.box_label(xyxy, label, color=colors(c, True))
            if save_crop:
                save_one_box(xyxy, imc, file=save_dir / "crops" / names[c] / f"{p.stem}.jpg", BGR=True)

        # Stream results
        im0 = annotator.result()
        view_img = True
        if view_img:
            if platform.system() == "Linux" and p not in windows:
                windows.append(p)
                cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
            cv2.imshow(str(p), im0)
            cv2.waitKey(1)  # 1 millisecond

        if save_img:
            if dataset.mode == "image":
                cv2.imwrite(save_path, im0)
            else:  # 'video' or 'stream'
                if vid_path[0] != save_path:  # new video
                    vid_path[0] = save_path
                    if isinstance(vid_writer[0], cv2.VideoWriter):
                        vid_writer[0].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix(".mp4"))  # force *.mp4 suffix on results videos
                    vid_writer[0] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                vid_writer[0].write(im0)


        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(obj_info) else '(no detections), '}")
    
    # After processing all frames, write the accumulated data to a single JSON file
    json_path = str(save_dir / "jsons" / "all_5th_frames.json")
    with open(json_path, 'w') as json_file:
    # Serialize and write each dictionary on a separate line
        for _, value in all_json_data.items():
            json_file.write(json.dumps(value) + '\n')

    # Print results location:
    LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")
    

def parse_opt():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default=ROOT / "data/images", help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--new_classification_file", type=str, help="csv with objects location and classes per frame")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    re_process_results(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
