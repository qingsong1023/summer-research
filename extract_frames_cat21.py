import os
import cv2

# 根目录和视频名称列表
ROOT_DIR = "D:/project/data_processes/data1"
VIDEO_NAMES = os.listdir(os.path.join(ROOT_DIR, "videos"))
VIDEO_NAMES = sorted([x for x in VIDEO_NAMES if 'mp4' in x])

# 将前16个视频作为训练集，后5个作为测试集
TRAIN_VIDEOS = VIDEO_NAMES[0:16]
TEST_VIDEOS = VIDEO_NAMES[16:22]

FRAME_NUMBERS = 0

def process_videos(video_names, save_dir_base):
    global FRAME_NUMBERS
    for video_name in video_names:
        print(video_name)
        vidcap = cv2.VideoCapture(os.path.join(ROOT_DIR, "videos", video_name))
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        print("fps", fps)
        if fps != 25:
            print(video_name, 'not at 25fps', fps)
        
        success = True
        count = 0
        save_dir = os.path.join(ROOT_DIR, save_dir_base, video_name.replace('.mp4', '') + '/')
        os.makedirs(save_dir, exist_ok=True)

        while success:
            success, image = vidcap.read()
            if success:
                # 保存每秒的帧
                if count % int(fps) == 0:
                    cv2.imwrite(save_dir + str(int(count // fps)).zfill(5) + '.png', image)
                count += 1

        vidcap.release()
        cv2.destroyAllWindows()
        print(count)
        FRAME_NUMBERS += count

# 处理训练视频
process_videos(TRAIN_VIDEOS, './frames/train/')

# 处理测试视频
process_videos(TEST_VIDEOS, './frames/test/')

print('Total Frames', FRAME_NUMBERS)