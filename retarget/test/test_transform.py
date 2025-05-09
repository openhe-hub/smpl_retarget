import torch
import numpy as np
from smplx import SMPLXLayer
from smplx.lbs import batch_rodrigues, batch_rigid_transform


def load_smplx_body_params(npz_path):
    data = np.load(npz_path)
    global_orient = torch.tensor(data['root_orient'], dtype=torch.float32).unsqueeze(1)  # (N, 1, 3)
    body_pose = torch.tensor(data['pose_body'], dtype=torch.float32).reshape(-1, 21, 3)          # (N, 21, 3)
    
        # Adjust betas to match the number of shape parameters in the SMPLX model
    num_betas = 16  # SMPLX model expects 16 shape parameters
    betas = torch.tensor(data.get('betas', np.zeros((global_orient.shape[0], 10))), dtype=torch.float32).unsqueeze(0)
    print(betas.shape)
    if betas.shape[1] < num_betas:
        padding = torch.zeros((betas.shape[0], num_betas - betas.shape[1]), dtype=torch.float32)
        betas = torch.cat([betas, padding], dim=1)  # Pad with zeros to match the required size
    
    print(global_orient.shape)
    print(body_pose.shape)
    print(betas.shape)
    return global_orient, body_pose, betas


def smplx_body_to_transforms(npz_path, model_folder):
    device = 'cpu'  # torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    global_orient, body_pose, betas = load_smplx_body_params(npz_path)
    N = global_orient.shape[0]

    full_pose = torch.cat([global_orient, body_pose], dim=1)  # (N, 22, 3)
    rot_mats = batch_rodrigues(full_pose.reshape(-1, 3)).reshape(N, 22, 3, 3).to(device)

    # Load SMPLX model to get joints and parents
    smplx_model = SMPLXLayer(model_path=model_folder, gender='NEUTRAL').to(device)
    with torch.no_grad():
        # Expand betas to match the batch size and perform the matrix multiplication
        betas_expanded = betas.unsqueeze(-1).unsqueeze(-1)  # Shape: (N, 10, 1, 1)
        shaped_vertices = smplx_model.v_template + (smplx_model.shapedirs[:, :, :10].permute(2, 0, 1) @ betas_expanded).squeeze(-1)
        joints = smplx_model.J_regressor[:22].to(device) @ shaped_vertices  # (N, 22, 3)

    # Get parent indices for first 22 joints
    parents = smplx_model.parents[:22].tolist()

    # Compute global transforms
    transforms, _ = batch_rigid_transform(rot_mats, joints.to(device), parents)

    return transforms  # shape: (N, 22, 4, 4)


# Example usage:
if __name__ == "__main__":
    npz_path = "/home/openhe/Workspace/program/python/smpl_retarget/retarget/retarget_input/B1_-_stand_to_walk_stageii.npz"
    model_folder = "/home/openhe/Workspace/program/python/smpl_retarget/retarget/human_model/smplx"  # <-- 指向含SMPLX模型的目录
    transforms = smplx_body_to_transforms(npz_path, model_folder)
    print("Transform shape:", transforms.shape)  
