from motion_lib_v2 import MotionLibV2

if __name__ == '__main__':
    motion_lib = MotionLibV2(
        motion_folder_path='./dataset',
        motion_cfg_path='./config/config.yaml',
        device='cuda'
    )