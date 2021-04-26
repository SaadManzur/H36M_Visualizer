import numpy as np

edges = [[0, 1], [0, 4], [0, 7], [1, 2], [2, 3], [4, 5], [5, 6], [7, 8], [8, 9], [9, 10], [7, 11], [11, 12], [12, 13]]
lefts = [4, 5, 6, 8, 9, 10]
rights = [1, 2, 3, 11, 12, 13]

def draw_skeleton(pose, ax, is_3d=False):

    col_right = 'b'
    col_left = 'r'

    if is_3d:
        ax.scatter(pose[:, 0], pose[:, 1], zs=pose[:, 2], color='k')
    else:
        ax.scatter(pose[:, 0], pose[:, 1], color='k')

    for u, v in edges:
        col_to_use = 'k'

        if u in lefts and v in lefts:
            col_to_use = col_left
        elif u in rights and v in rights:
            col_to_use = col_right

        if is_3d:
            ax.plot([pose[u, 0], pose[v, 0]], [pose[u, 1], pose[v, 1]], zs=[pose[u, 2], pose[v, 2]], color=col_to_use)
        else:
            ax.plot([pose[u, 0], pose[v, 0]], [pose[u, 1], pose[v, 1]], color=col_to_use)

def correct_pose(pose, lower_hip, hip_len_mm=None):

    hip, rhip, lhip = lower_hip
    configs = [[hip, rhip], [hip, lhip]]

    mid_point = (pose[lhip] + pose[rhip])/2.0
    hip_to_mhip = mid_point - pose[hip]
    hip_to_mhip_u = hip_to_mhip / np.linalg.norm(hip_to_mhip + 1e-10) # in a rare case when mhip and hip is the same point

    for hip, xhip in configs:

        hip_to_xhip = pose[xhip] - pose[hip]
        hip_to_chip = (hip_to_xhip - hip_to_mhip)
        hip_to_chip /= np.linalg.norm(hip_to_chip)

        pose[xhip] = pose[xhip] - np.linalg.norm(hip_to_mhip) * hip_to_mhip_u

        hip_to_xhip_n = pose[xhip] - pose[hip]
        hip_to_xhip_nu = hip_to_xhip_n / np.linalg.norm(hip_to_xhip_n)

        multiplication_factor = np.linalg.norm(hip_to_xhip_n)

        if hip_len_mm is not None:
            multiplication_factor = hip_len_mm

        pose[xhip] = pose[hip] + hip_to_xhip_nu*multiplication_factor

    return pose

def get_angles_from_joints(joints, parents, names, bone_names):
    
    joint_angles = [np.zeros(3)]*(joints.shape[0]-1)
    
    for i in range(joints.shape[0]):
        parent = parents[i]
        
        if parent >= 0:
            vec = joints[i]-joints[parent]
            vec /= np.linalg.norm(vec)
            name = f"{names[parent]}_{names[i]}"
            idx = bone_names.index(name)
            angles = [np.arccos(vec[0]), np.arccos(vec[1]), np.arccos(vec[2])]
            joint_angles[idx] = angles
    
    return np.array(joint_angles)

def get_joints_from_angles(angles, adjacency, names, bone_names, bone_lengths):

    queue = []

    queue.append(0)

    joints = []

    for i in range(angles.shape[0]+1):
        joints.append(np.zeros(3))

    while len(queue) > 0:
        current = queue.pop(0)

        for child in adjacency[current]:
            queue.append(child)
            name = names[current] + "_" + names[child]
            angle = np.array(angles[bone_names.index(name)])
            vec = np.array([np.cos(angle[0]), np.cos(angle[1]), np.cos(angle[2])])
            joints[child] = bone_lengths[name]*vec + joints[current]
    
    return np.array(joints)