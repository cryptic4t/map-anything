import os
from tabnanny import verbose 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch

from mapanything.models import MapAnything
from mapanything.utils.image import load_images

import numpy as np
import open3d as o3d


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


model = MapAnything.from_pretrained("facebook/map-anything").to(device)

images = "bottle video/"
views = load_images(images) 


predictions = model.infer(
    views,
    memory_efficient_inference=False,
    use_amp=True,
    mask_edges=True,
    apply_confidence_mask=False,
    confidence_percentile=10,
)

for i, pred in enumerate(predictions):

    #geometry outputs
    pts3d = pred["pts3d"]
    pts3d_cam = pred["pts3d_cam"]
    depth_z = pred["depth_z"]
    depth_along_ray = pred["depth_along_ray"]

    #camera outputs
    ray_directions = pred["ray_directions"]
    intrinsics = pred["intrinsics"]
    camera_poses = pred["camera_poses"]
    cam_trans = pred["cam_trans"]
    cam_quats = pred["cam_quats"]

    #quality and masking
    confidence = pred["conf"]
    mask = pred["mask"]
    non_ambiguous_mask = pred["non_ambiguous_mask"]
    non_ambiguous_mask_logits = pred["non_ambiguous_mask_logits"]

    #scaling
    metric_scaling_factor = pred["metric_scaling_factor"]

    #original input
    img_no_norm = pred["img_no_norm"]

np_pts3d = np.array(pts3d.cpu())

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np_pts3d[0][1])
o3d.visualization.draw_geometries([pcd])

def extract_points_from_prediction(prediction, apply_mask):
    """Extract valid 3D points and mask from a single view prediction."""
    pts3d = prediction["pts3d"].cpu().numpy()

    if pts3d.ndim == 4:
        pts3d = pts3d[0]

    if apply_mask and "mask" in prediction:
        mask = prediction["mask"].cpu().numpy()
        if mask.ndim == 4:
            mask = mask[0, :, :, 0]  # Assuming mask has shape [1, H, W, 1]
        elif mask.ndim == 3:
            mask = mask[0]
        mask_bool = mask > 0.5
    else:
        mask_bool = np.ones((pts3d.shape[0], pts3d.shape[1]), dtype=bool)

    points = pts3d[mask_bool]
    return points, mask_bool

#test on first view
points_view0, mask_view0 = extract_points_from_prediction(predictions[0], apply_mask=True)
print(f"View 0: Extracted {points_view0.shape[0]:,} valid points.")

#visualize single view
pcd_view0 = o3d.geometry.PointCloud()
pcd_view0.points = o3d.utility.Vector3dVector(points_view0)
pcd_view0.paint_uniform_color([0.7, 0.3, 0.3])  
print("Visualizing first view only...")
o3d.visualization.draw_geometries([pcd_view0], window_name="View 0 Only")

def extract_colors_from_prediction(prediction, mask_indices):
    """Extract RGB colors for valid points from the input image"""
    img = prediction["img_no_norm"].cpu().numpy()
    
    if img.ndim == 4:
        img = img[0]
    
    colors = img[mask_indices]
    colors = np.clip(colors, 0, 1)
    return colors

#apply colors to first view 
colors_view0 = extract_colors_from_prediction(predictions[0], mask_view0)
pcd_view0.colors = o3d.utility.Vector3dVector(colors_view0)
print(f"Applied {colors_view0.shape[0]:,} colors to view 0.")

#visualize with colors
print("Visualizing first view with colors...")
o3d.visualization.draw_geometries([pcd_view0], window_name="View 0 with Colors")

def merge_all_views_to_pointcloud(predictions, apply_mask=True, verbose=True):
    """Merge point clouds from all views into a single Open3D point cloud with RGB colors."""
    all_points = []
    all_colors = []

    for i, pred in enumerate(predictions):
        points, mask = extract_points_from_prediction(pred, apply_mask=apply_mask)
        colors = extract_colors_from_prediction(pred, mask)

        all_points.append(points)
        all_colors.append(colors)

        if verbose:
            print(f"View {i+1}/{len(predictions)}: Extracted {points.shape[0]:,} points and {colors.shape[0]:,} colors.")    

    merged_points = np.vstack(all_points)
    merged_colors = np.vstack(all_colors)

    if verbose:
        print(f"\n Total merged points: {merged_points.shape[0]:,}")

    # Create a new point cloud
    pcd_merged = o3d.geometry.PointCloud()
    pcd_merged.points = o3d.utility.Vector3dVector(merged_points)
    pcd_merged.colors = o3d.utility.Vector3dVector(merged_colors)

    return pcd_merged

# Merge all views into a single point cloud
pcd_complete = merge_all_views_to_pointcloud(predictions, apply_mask=True, verbose=True)

#Visualize the complete point cloud
print("Visualizing complete point cloud...")
o3d.visualization.draw_geometries([pcd_complete], window_name="Complete Reconstruction")

# Save the complete point cloud
o3d.io.write_point_cloud(
    "bottle/results/reconstruction.ply",
    pcd_complete
)