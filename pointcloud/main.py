#!/usr/bin/env python3
"""
Implementation of the RANSAC (RAndom SAmple Consensus) algorithm for finding outliers in 3D space.

Algorithm:
    1. Random select a (minimal) subset (3 points to form plane)
    2. Instantiate the model (a plane) from it
    3. Using the model, classify all of the data points as inliers or outliers
        (use some threshold + euclidean distance)
    4. Repeat steps 1-3 for N iterations
    5. Return the LARGEST inlier set to form the inlier/outlier split

This technique can be used for obstacle detection from point cloud data by finding the ground plane 
with the obstacles being the outliers.

Source: https://youtu.be/5E5n7fhLHEM

@author Jacob Bringham
@version 4/11/2022
"""

import random
import re
import numpy as np
import matplotlib.pyplot as plt
import time
import open3d as o3d
import pptk

def readCoordinates(filename):
    """
    Read in a text file containing lines of x, y, z coordinates and create a 2D numpy array.

    There's probably some numpy method for this but :shrug:

    @param filename text file containing lines of x, y, z coordinates
    @return 2D Numpy array of X, Y, Z values
    """
    with open(filename, "r") as fp:
        points = []

        lines = fp.readlines()

        for line in lines:
            tokens = re.split(r",\s*", line.strip())
            points.append([float(token) for token in tokens])

        return np.array(points)
        
    raise ValueError(f"Failed to read {filename}")


def voxelFilter(data, voxel_size):
    """
    Reduces the overall amount of points by slicing the world into voxels (cubes)
    and averaging all of the points within that voxel.

    @param data       2D array of X, Y, Z coordinates
    @param voxel_size Size of the cube to slice the world
    @return           Smaller 2D array of X, Y, Z coordinates
    """



def RANSAC(data, dist_threshold, max_iterations):
    """
    Performs the RANSAC algorithm on a 2D array of X, Y, Z coordinates (Nx3).

    Attempts to fit a plane through the points and return the points near
    that plane separated from the rest of the data.

    @param data            2D array of X, Y, Z coordinates
    @param dist_threshold  Distance to fitted plane to be considered an inlier
    @param max_iterations  Number of iterations to run RANSAC for
    @return (inliers, outliers)
    """

    global_inliers  = np.array([])
    global_outliers = np.array([])

    for i in range(max_iterations):
        
        # Select 3 random points to form a plane
        chosen = random.sample(population=range(0, data.shape[0]), k=3)
        x1, y1, z1 = data[chosen[0]]
        x2, y2, z2 = data[chosen[1]]
        x3, y3, z3 = data[chosen[2]]

        # Plane Equation --> Ax + By + Cz + D = 0
        # Value of constants for sampled inlier plane
        a = (y2 - y1)*(z3 - z1) - (z2 - z1)*(y3 - y1)
        b = (z2 - z1)*(x3 - x1) - (x2 - x1)*(z3 - z1)
        c = (x2 - x1)*(y3 - y1) - (y2 - y1)*(x3 - x1)
        d = -(a*x1 + b*y1 + c*z1)
        
        # For every point in the dataset, calculate the distance from the plane
        # |Ax + By + Cz + D| / √(A^2 + B^2 + C^2)
        denom     = np.sqrt(a*a + b*b + c*c)
        distances = np.abs( (a * data[:, 0]) + (b * data[:, 1]) + (c * data[:, 2]) + d ) / denom
        distances = distances[:, np.newaxis]
       
        # Find all of the inliers
        inliers = data[(distances <= dist_threshold)[:, 0], :]

        # If this is the largest inlier set we've seen, save it!
        if inliers.shape[0] > global_inliers.shape[0]:
            global_inliers  = inliers
            global_outliers = data[(distances > dist_threshold)[:, 0], :]

    return global_inliers, global_outliers




if __name__ == "__main__":
    t1 = time.time()
    coords = readCoordinates("pcl_8879.txt")
    t2 = time.time()
    print(f"readCoordinates() took {t2 - t1} seconds!")

    if coords.shape[0] < 100:
        print(coords)
    print(coords.shape)
    print()


    for trial in range(1):
        t1 = time.time()
        inliers, outliers = RANSAC(coords, 2, 100)
        t2 = time.time()
        print(f"RANSAC took {t2 - t1} seconds!")

        print(inliers.shape)
        print(outliers.shape)

    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.scatter3D(coords[:, 0], coords[:, 1], coords[:, 2], cmap='Greens');
    # plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(inliers[:, 0],  inliers[:, 1],  inliers[:, 2],  marker='o', c="black")
    ax.scatter(outliers[:, 0], outliers[:, 1], outliers[:, 2], marker='o', c="red")
    # for ii in range(-100,100,10):
        # ax.view_init(elev=10., azim=180)
        # plt.savefig("view_180_%d.png" % ii)
    # ax.view_init(elev=10., azim=180)
    plt.show()
    
    # v = pptk.viewer(coords.transpose())

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(coords)
    # pcd.colors = o3d.utility.Vector3dVector(np.ones(coords.shape))
    # pcd.colors = o3d.utility.Vector3dVector(colors/65535)
    # pcd.normals = o3d.utility.Vector3dVector(normals)
    # o3d.visualization.draw_geometries([pcd])
    
