import os
import numpy as np
import open3d as o3d
from tqdm import tqdm

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-data_path", type=str, default="/tmp-network/dataset/HO3D_V2/train",
    help="Path to HO3D dataset train or val or test directory", required=False)
    ap.add_argument("-save_path", type=str, default="/tmp-network/dataset/HO3D_V2/temp",
    help="Path to save extents", required=False)

    args = vars(ap.parse_args())
    
    baseDir = args['data_path']
    saveDir = args['save_path']
    bbox_extents_list = []
    for seq in tqdm(os.listdir(baseDir)):
            bbox_extents_seq = []
            # print(os.path.join(baseDir, seq, 'meta'))
            for filename in tqdm(os.listdir(os.path.join(baseDir, seq, 'mesh_watertight', 'hand'))):
                if os.path.splitext(filename)[1] == '.off':
                    #print("obj mesh path:", os.path.join(baseDir, seq, 'mesh', 'obj', filename))
                    #print("hand mesh path:", os.path.join(baseDir, seq, 'mesh_watertight', 'hand', filename))
                    o_m = o3d.io.read_triangle_mesh(os.path.join(baseDir, seq, 'mesh', 'obj', filename))
                    h_m = o3d.io.read_triangle_mesh(os.path.join(baseDir, seq, 'mesh_watertight', 'hand', filename))
                    h_o_m = o_m + h_m
                    #print("combined mesh:", h_o_m)
                    aabb = h_o_m.get_axis_aligned_bounding_box()
                    bbox_extents_seq.append(aabb.get_extent())
                    bbox_extents_list.append(aabb.get_extent())
                    #print("bbox_extents_list", bbox_extents_list)
            print("completed  seq:", seq)
            print("saving... check in directory:", saveDir)
            with open(os.path.join(saveDir, seq+'.npy'), 'wb') as f:
                np.save(f, bbox_extents_seq)
    print("comleted all seqs")
    print("saving... check in directory:", saveDir)
    with open(os.path.join(saveDir, 'all_bbox_extents.npy'), 'wb') as f:
        np.save(f, bbox_extents_list)
    print("bbox_extents_list:", bbox_extents_list)