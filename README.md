### Retarget
1. Download SMPLX models under retarget/human_model/smplx, or download SMPL models under retarget/human_models/smpl
2. Prepare SMPL/SMPLX dataset (*.npz) into retarget/retarget_input
3. Run retarget_motion_smplx.py
### MotionLib
1. Set motion data info: motion_lib/config/config.yaml
2. Prepare retargeted data (*.npy / *.pkl) under motion_lib/dataset
3. Run test.py for demo
4. Use API: motion_lib/motion_lib_v2.py
5. Mind: I also put motion_lib.py from 'expressive-humanoid' for comparison
### Pipeline Migration
* Origin FBX pipeline in 'poselib' & 'expressive-humanoid':
  1. *.fbx dataset 
  2. FBX SDK: *.npy dataset
  3. Retarget in 'expressive-humanoid': *.npy dataset(SkeletonMotion)
  4. Train in 'legged_gym': 
     1. Motion Lib import: *.npy(SkeletonMotion) dataset => motion_lib
     2. Use: motion_lib => motion data (frames, fps, SkeletonState, ...)
* Curr SMPL/SMPLX pipeline:
  1. *.npz SMPL/SMPLX dataset 
  2. Retarget in 'UH-1': *.npy dataset(origin numpy)
  3. Train in 'legged_gym': 
     1. Motion Lib import: *.npy(origin numpy) dataset => motion_lib_v2
     2. Use: motion_lib_v2 => motion data (frames, fps, SkeletonState, ...)