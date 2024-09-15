import numpy as np
import os
import cv2
import pickle
import pandas as pd 
from tqdm import tqdm

def main():
    ROOT_DIR = "D:\project\data_processes\data1"
    VIDEO_DIR = os.path.join(ROOT_DIR, 'videos')
    VIDEO_NAMES = os.listdir(VIDEO_DIR)
    VIDEO_NAMES = sorted([x for x in VIDEO_NAMES if x.endswith('.mp4')])
    
    TRAIN_NUMBERS = [f'case_{str(i).zfill(2)}' for i in range(0, 16)]  
    TEST_NUMBERS = [f'case_{str(i).zfill(2)}' for i in range(16, 22)]  

    TRAIN_FRAME_NUMBERS = 0
    TEST_FRAME_NUMBERS = 0

    train_pkl = dict()
    test_pkl = dict()

    unique_id = 0
    unique_id_train = 0
    unique_id_test = 0

    phase2id = {
        'not_initialized': 0,
        'Incision': 1,
        'Viscoelasticum': 2,
        'Rhexis': 3,
        'Hydrodissektion': 4,
        'Phako': 5,
        'Irrigation-Aspiration': 6,
        'Kapselpolishing': 7,
        'Linsenimplantation': 8,
        'Visco-Absaugung': 9,
        'Tonisieren': 10,
        'Antibiotikum': 11
    }

    for video_name in VIDEO_NAMES:
        video_id = video_name.replace('.mp4', '')
        
        if video_id in TRAIN_NUMBERS:
            unique_id = unique_id_train
        elif video_id in TEST_NUMBERS:
            unique_id = unique_id_test
        else:
            print(f"Skipping {video_name}: Not in train or test set.")
            continue

        vidcap = cv2.VideoCapture(os.path.join(VIDEO_DIR, video_name))
        
        if not vidcap.isOpened():
            print(f"Error: Cannot open video {video_name}")
            continue

        fps = vidcap.get(cv2.CAP_PROP_FPS)
        if fps != 25:
            print(video_name, 'not at 25fps', fps)
        
        frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)

        phase_path = os.path.join(VIDEO_DIR, video_id + '.csv')
        if not os.path.exists(phase_path):
            print(f"Error: CSV file {phase_path} does not exist.")
            continue

        phase_df = pd.read_csv(phase_path)

        print("CSV Columns:", phase_df.columns)

        phase_dict = phase_df.set_index(phase_df.columns[0])[phase_df.columns[1]].to_dict()

        frame_infos = list()
        frame_id_ = 0
        for frame_id in tqdm(range(0, int(frames))):
            if frame_id % fps == 0:
                info = dict()
                info['unique_id'] = unique_id
                info['frame_id'] = frame_id % fps
                info['video_id'] = video_id

                if frame_id in phase_dict:
                    phase_name = phase_dict[frame_id]
                    phase_id = phase2id.get(phase_name, None)
                    info['phase_gt'] = phase_id
                    info['phase_name'] = phase_name
                else:
                    info['phase_gt'] = None
                    info['phase_name'] = None

                info['fps'] = 1
                info['original_frames'] = int(frames)
                info['frames'] = int(frames) // fps
                frame_infos.append(info)
                unique_id += 1
                frame_id_ += 1
        
        if video_id in TRAIN_NUMBERS:
            train_pkl[video_id] = frame_infos
            TRAIN_FRAME_NUMBERS += frames
            unique_id_train = unique_id
        elif video_id in TEST_NUMBERS:
            test_pkl[video_id] = frame_infos
            TEST_FRAME_NUMBERS += frames
            unique_id_test = unique_id

    labels_dir = os.path.join(ROOT_DIR, 'labels')
    train_save_dir = os.path.join(labels_dir, 'train')
    os.makedirs(train_save_dir, exist_ok=True)
    with open(os.path.join(train_save_dir, '1fpstrain.pickle'), 'wb') as file:
        pickle.dump(train_pkl, file)

    test_save_dir = os.path.join(labels_dir, 'test')
    os.makedirs(test_save_dir, exist_ok=True)
    with open(os.path.join(test_save_dir, '1fpsval_test.pickle'), 'wb') as file:
        pickle.dump(test_pkl, file)

    print('TRAIN Frames', TRAIN_FRAME_NUMBERS, unique_id_train)
    print('TEST Frames', TEST_FRAME_NUMBERS, unique_id_test)

if __name__ == '__main__':
    main()
