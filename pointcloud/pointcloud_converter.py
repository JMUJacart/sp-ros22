#!/usr/bin/env python
# coding: utf-8

"""
This ROS Node uses RANSAC and DBSCAN to locate obstacles from PointCloud2 data.

As the data comes off of the camera, it should be filtered down somehow or set 
to a low resolution. Each PointCloud can have >1,000,000 xyz coordinates that 
are all extremely close to each other (<0.001) which can hurt the runtime 
of this node which needs to process a PointCloud in anywhere from 0.1Hz 
(10s per PointCloud) to 100Hz(0.001s per PointCloud).

After the data comes in, RANSAC (RANdom Sampling And Consensus) is used in
order to fit a plane through the cloud which *should* be the ground. If the
"ground" plane happens to be upright enough to look like a wall, we should
average all of the points and report that an obstacle is ahead.

Next, we can remove all of the ground points and cluster the rest of the data using
DBSCAN (Density-based spatial clustering of applications with noise). 
If there are a bunch of points in the outlier set this can take a long time
since this algorithm is O(N^2) in the worst case. Consider filtering down
the outliers even more if this becomes a problem.


Sidenote:
    If you'd like to test this node with a bag file (recommended) and
    you can't localize for whatever reason (LiDAR not plugged in, etc.)
    then run this command to add a dummy tf so you can visualize the cloud.

rosrun tf static_transform_publisher 0 0 0 0 0 0 map front_cam_camera_center 100

@author  Jacob Bringham
@version 4/19/2022
"""

import random
import rospy
import numpy as np
import math
import struct
import time
import tf
import tkinter as tk

from sklearn.cluster import DBSCAN

# Messages
from navigation_msgs.msg import VehicleState, Obstacle, ObstacleArray
from geometry_msgs.msg import TwistStamped, Vector3, PointStamped
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2

# Display
from visualization_msgs.msg import Marker

PointFieldDataTypes = {
    1: "INT8",
    2: "UINT8",
    3: "INT16",
    4: "UINT16",
    5: "INT32",
    6: "UINT32",
    7: "FLOAT32",
    8: "FLOAT64"
}

ObjectColorMap = {
    -1: {"r": 0.50, "g": 0.50, "b": 0.50},
     0: {"r": 0.50, "g": 0.00, "b": 0.75},
     1: {"r": 0.25, "g": 0.00, "b": 0.80},
     2: {"r": 0.00, "g": 1.00, "b": 1.00},
     3: {"r": 1.00, "g": 1.00, "b": 0.00},
     4: {"r": 1.00, "g": 1.00, "b": 0.50},
}

class PointCloudObstacleDetector(object):
    
    def __init__(self):
        
        # ----- Parameters -----
        self.name        = rospy.get_param("name", "converter")
        self.cloud_in    = rospy.get_param("cloud_in", "/voxel_grid/output") # "/front_cam/front/point_cloud/cloud_registered"
        self.cloud_out     = rospy.get_param("cloud_out",    "/object_cloud")
        self.ransac_out    = rospy.get_param("ransac_out",   "/ransac_cloud")
        self.obstacle_out  = rospy.get_param("obstacle_out", "/ransac_obstacles")
        self.tuning        = rospy.get_param("tuning", True)
        # RANSAC Parameters
        self.dist_threshold = rospy.get_param("dist_threshold", 0.45) # Starships are ~0.55m tall
        self.max_iterations = rospy.get_param("max_iterations", 35)
        # Outliers are voxelized before they are clustered
        self.filter_size = rospy.get_param("filter_size", 0.75)
        # DBSCAN Parameters
        self.eps         = rospy.get_param("eps", self.filter_size * 2)
        self.min_samples = rospy.get_param("min_samples", 6.0)
        rospy.init_node(self.name)
        
        # ----- Node State -----
        self.cloud_sub    = rospy.Subscriber(self.cloud_in,    PointCloud2, self.findClusters)
        self.cloud_pub    = rospy.Publisher(self.cloud_out,    PointCloud2, queue_size=10)
        self.ransac_pub   = rospy.Publisher(self.ransac_out,   PointCloud2, queue_size=10)
        self.obstacle_pub = rospy.Publisher(self.obstacle_out, ObstacleArray, queue_size=10)
        self.display_pub  = rospy.Publisher(self.obstacle_out + "_markers", Marker, queue_size=10)
        self.obstacles = ObstacleArray()

        if self.tuning:
            self.tuner = ParameterTunerGUI(
                self.dist_threshold, \
                self.max_iterations, \
                self.filter_size, \
                self.eps, 
                self.min_samples \
            )

            self.tuner.window.mainloop()

        r = rospy.Rate(1)
        while not rospy.is_shutdown():
            r.sleep()


    def updateParams(self):
        """
        Updates the parameters of this node based on the values from the GUI.
        """
        if self.tuner.check():
            dist, it, vox, eps, samp = self.tuner.retrieve()
            self.dist_threshold = dist
            self.max_iteration = it
            self.filter_size = vox
            self.eps = eps
            self.min_samples = samp

            print("Updated to %s" % (str(self.tuner.retrieve())))


    def printCloud(self, msg):
        """
        Prints out information about a PointCloud, not including the data.

        @param msg  Incoming PointCloud2 message
        """
        print("Width x Height: %dx%d" % (msg.width, msg.height))  

        for i, pf in enumerate(msg.fields):
            print("Name: %s\nOffset: %u\nDatatype: %s\nCount: %d\n" % \
                 (pf.name, pf.offset, PointFieldDataTypes[pf.datatype], pf.count))

        print("BigEndian: %s" % (str(msg.is_bigendian)))
        print("PointStep: %u" % (msg.point_step))
        print("RowStep: %u" % (msg.row_step))
        print("IsDense: %s" % (str(msg.is_dense)))


    def findClusters(self, msg):
        """
        Finds the clusters of obstacles within the pointcloud by receiving a (hopefully) filtered
        pointcloud, performing RANSAC on it to locate the ground plane and then performing DBSCAN
        to locate the obstacle clusters after the ground plane is removed.

        Default ZED2i behaviour sends a PointCloud at around 10Hz, so this needs to be performant.

        @param msg  Incoming PointCloud2 message
        """

        s = time.time()
        if self.tuning:
            self.updateParams()

        self.obstacles = ObstacleArray()
        self.obstacles.header = msg.header

        # Convert the PointCloud into a 2D numpy array of (x, y, z, rgba)
        data = np.fromiter(pc2.read_points(cloud=msg, field_names=None, skip_nans=False, uvs=[]), \
            dtype="f4,f4,f4,f4").view(np.float32).reshape(-1, 4)
        
        # Strip the color info (can contain NaN's!)
        data = np.delete(data, 3, axis=1)

        # x = x[~numpy.isnan(x).any(axis=1)] # Removes NaN's manually
        
        # Locate the ground plane via RANSAC (the inliers)
        inliers, outliers = self.RANSAC(data, self.dist_threshold, self.max_iterations)

        # Publish a PointCloud of the inliers and outliers for debugging
        self.toPointCloud2(inliers, outliers, msg.header)

        if outliers.shape[0] > 0:

            # Downsample the outliers so DBSCAN can be quicker
            filtered_outliers = self.voxelize(outliers, self.filter_size)

            # Cluster the points using DBSCAN
            model = DBSCAN(eps=self.eps, min_samples=self.min_samples)
            pred = model.fit_predict(filtered_outliers)

            obstacles = np.asarray(np.unique(pred)).T
            pred.shape = (pred.shape[0], 1)

            # Publish a PointCloud relating points to obstacles for debugging
            cluster_cloud = np.array([])
            for cluster in obstacles:
                # Mask out all of the data points for this cluster
                mask = pred == cluster
                mask = np.column_stack((mask, mask, mask))
                c = filtered_outliers[mask].reshape(-1, 3)
                
                # Average all of the points and create an obstacle if not noise
                if cluster != -1:
                    obs = Obstacle()
                    obs.header = msg.header
                    obs.pos.header = msg.header
                    obs.pos.point.x = np.min(c[:, 0], axis=0)
                    obs.pos.point.y = np.average(c[:, 1], axis=0)
                    obs.pos.point.z = np.average(c[:, 2], axis=0)
                    obs.radius = 0.25
                    self.obstacles.obstacles.append(obs)

                # ----- DEBUGGING -----
                # Publish a cloud of points with a unique color per cluster
                if not cluster in ObjectColorMap:
                    ObjectColorMap[cluster] = {"r": random.random(), "g": random.random(), "b": random.random()}

                color = ObjectColorMap[cluster]
                if cluster_cloud.size == 0:
                    cluster_cloud = self.addColor(c, color["r"], color["g"], color["b"], 1)
                else:
                    cluster_cloud = np.concatenate( (cluster_cloud, self.addColor(c, color["r"], color["g"], color["b"], 1)), axis=0)

            cloud_out = self.point_cloud(cluster_cloud, msg.header)
            self.cloud_pub.publish(cloud_out)
            # ----- DEBUGGING -----

        self.obstacle_pub.publish(self.obstacles)
        self.local_display()
        # print("Cloud #%d took %f seconds!" % (msg.header.seq, time.time() - s))
        self.tuner.update(time.time() - s)
                
               


    def RANSAC(self, data, dist_threshold, max_iterations):
        """
        Performs the RANSAC algorithm on a 2D array of X, Y, Z coordinates (Nx3).

        Attempts to fit a plane through the points and return the points near
        that plane separated from the rest of the data.

        @param data            2D array of X, Y, Z coordinates
        @param dist_threshold  Distance to fitted plane to be considered an inlier
        @param max_iterations  Number of iterations to run RANSAC for
        @return (inliers, outliers)
        """

        # 3 points are needed to fit a plane
        if data.shape[0] <= 3:
            rospy.logwarn("[%s] RANSAC received only %d points!" % (self.name, data.size))
            return data, np.array([])

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
            # |Ax + By + Cz + D| / sqrt(A^2 + B^2 + C^2)
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


    def voxelize(self, data, size, avg=False):
        """
        Discretizes 3D space into cubes called "voxels". All points located within
        the same voxel are combined into the point at the center of their voxel.

        You can also average together all points within a voxel, but it is recommended
        that you use a library for that as it will perform better.

        See: http://wiki.ros.org/pcl_ros/Tutorials/VoxelGrid%20filtering

        @param self 
        @param data Nx3 numpy array of coordinates
        @param size Length of one side of a voxel cube 
        """
        if not avg:
            return np.unique(data // size, axis=0) * size + (size / 2)
        else:
            voxels = data // size



    def addColor(self, data, r, g, b, a):
        """
        Appends 4 extra columns to data so the resulting data can be a colored PointCloud.

        @param data  Nx3 numpy array of (x, y, z)
        @param r     Value to add to the red channel [0, 1]
        @param g     Value to add to the green channel [0, 1]
        @param b     Value to add to the blue channel [0, 1]
        @param a     Value to add to the alpha channel [0, 1]
        @return data Nx7 numpy array of (x, y, z, r, g, b, a)
        """
        tmp = data

        # Append color information to the array
        tmp = np.hstack( (tmp, np.full( (tmp.shape[0], 1), float(r) )) )
        tmp = np.hstack( (tmp, np.full( (tmp.shape[0], 1), float(g) )) )
        tmp = np.hstack( (tmp, np.full( (tmp.shape[0], 1), float(b) )) )
        tmp = np.hstack( (tmp, np.full( (tmp.shape[0], 1), float(a) )) )

        return tmp


    def toPointCloud2(self, inlier, outlier, header):
        """
        Converts the inlier data and the outlier data into a singular PointCloud2
        that can be visualized.

        @param inlier  Inlier data set (GREEN)
        @param outlier Outlier data set (RED)
        @param header  Header of the PointCloud
        """
        # Append color information to the arrays
        inlier = self.addColor(inlier, 0, 1, 0, 1)
        outlier = self.addColor(outlier, 1, 0, 0, 1)

        merged = np.concatenate((inlier, outlier), axis=0)
        merged = merged.astype(np.float32)

        cloud_out = self.point_cloud(merged, header)
        self.ransac_pub.publish(cloud_out)
        

    def point_cloud(self, points, header):
        """ 
        Creates a point cloud message from an Nx7 array of (x, y, z, r, g, b, a)
        where r, g, b, a are floats in the range [0, 1].

        @param points  Nx7 array of xyz positions (m) and rgba colors (0..1)
        @param header  Header tag to add to the cloud
        @return PointCloud2 message
        """
        ros_dtype = PointField.FLOAT32
        dtype = np.float32
        itemsize = np.dtype(dtype).itemsize

        data = points.astype(dtype).tobytes()

        fields = [PointField(
            name=n, offset=i*itemsize, datatype=ros_dtype, count=1)
            for i, n in enumerate('xyzrgba')]


        return PointCloud2(
            header=header,
            height=1,
            width=points.shape[0],
            is_dense=True,
            is_bigendian=False,
            fields=fields,
            point_step= (itemsize * 7),
            row_step=(itemsize * 7 * points.shape[0]),
            data=data
        )

    def local_display(self):
        """
        Publish markers for RViz to display, may appear jittery due to lack of interpolation.

        @param self
        """
        object_list = self.obstacles.obstacles
        for i in range(len(object_list)):
            marker = Marker()
            marker.header = Header()
            # rospy.logwarn("[%s] I have an obstacle in frame '%s'!" % (self.name, object_list[i].header.frame_id))
            marker.header.frame_id = object_list[i].header.frame_id

            color = ObjectColorMap[i]
    
            marker.ns = "Object_NS"
            marker.id = i
            marker.type = Marker.CYLINDER
            marker.action = 0
            marker.color.r = color["r"] # 1.0
            marker.color.g = color["g"] # 1.0
            marker.color.b = color["b"] # 0.0
            marker.color.a = 1.0
            marker.lifetime = rospy.Duration.from_sec(0.50)

            marker.pose.position.x = object_list[i].pos.point.x
            marker.pose.position.y = object_list[i].pos.point.y
            marker.pose.position.z = 0.0

            radius = object_list[i].radius
            marker.scale.x = radius
            marker.scale.y = radius
            marker.scale.z = 0.1

            self.display_pub.publish(marker)



class ParameterTunerGUI():
    def __init__(self, dist, max_iter, vox, eps, min_sample):
        """
        Initializes a GUI used to tune the parameters that process the PointCloud.
        @param self
        @param dist     Initial value
        @param max_iter Initial Value
        @param vox      Initial Value
        @param eps      Initial Value
        @param sample   Initial Value
        """

        self.params = {
            "dist":       float(dist),
            "max_iter":   float(max_iter),
            "vox":        float(vox),
            "eps":        float(eps),
            "min_sample": float(min_sample),
        }

        self.window = tk.Tk()
        self.window.winfo_toplevel().title("PointCloud Parameter Tuner")

        # Make labels
        self.dist_lbl = tk.Label(self.window, text="Distance Threshold")
        self.iter_lbl = tk.Label(self.window, text="Max Iterations")
        self.vox_lbl  = tk.Label(self.window, text="Filter Size")
        self.eps_lbl  = tk.Label(self.window, text="Epsilon")
        self.samp_lbl = tk.Label(self.window, text="Min Samples")

        self.dist_lbl.grid(row=0)
        self.iter_lbl.grid(row=1)
        self.vox_lbl .grid(row=2) 
        self.eps_lbl .grid(row=3)
        self.samp_lbl.grid(row=4)

        # Make textboxes
        self.dist_var = tk.StringVar(self.window, str(dist))
        self.iter_var = tk.StringVar(self.window, str(max_iter))
        self.vox_var  = tk.StringVar(self.window, str(vox))
        self.eps_var  = tk.StringVar(self.window, str(eps))
        self.samp_var = tk.StringVar(self.window, str(min_sample))

        self.dist_ent = tk.Entry(textvariable=self.dist_var)
        self.iter_ent = tk.Entry(textvariable=self.iter_var)
        self.vox_ent  = tk.Entry(textvariable=self.vox_var)
        self.eps_ent  = tk.Entry(textvariable=self.eps_var)
        self.samp_ent = tk.Entry(textvariable=self.samp_var)

        self.dist_ent.grid(row=0, column=1)
        self.iter_ent.grid(row=1, column=1)
        self.vox_ent .grid(row=2, column=1)
        self.eps_ent .grid(row=3, column=1)
        self.samp_ent.grid(row=4, column=1)

        self.runtime_var = tk.StringVar(self.window, "0.0 Clouds / Second")
        self.runtime_lbl = tk.Label(self.window, textvariable=self.runtime_var).grid(row=5)

        self.enter_btn = tk.Button(self.window, text="Update", command=self.submit).grid(row=5, column=1)
        self.hasUpdate = False

    def update(self, runtime):
        """
        Updates the display with new information.
        @param runtime  Time elapsed for most recent PointCloud calc
        """
        if runtime > 0:
            self.runtime_var.set("%6.1f Clouds / Second" % (1 / runtime))

    def submit(self):
        """
        Set values to be grabbed for the next PointCloud.
        """
        def handle(key, var, ent):
            try:
                self.params[key] = float(var.get())
                ent.configure({"foreground": "black"})
            except:
                ent.configure({"foreground": "red"})

        handle("dist", self.dist_var, self.dist_ent)
        handle("iter", self.iter_var, self.iter_ent)
        handle("vox",  self.vox_var,  self.vox_ent)
        handle("eps",  self.eps_var,  self.eps_ent)
        handle("samp", self.samp_var, self.samp_ent)
        self.hasUpdate = True
        print("submit() - %s" % (self.params))

    def retrieve(self):
        """
        Get the values associated with each param.
        """
        self.hasUpdate = False
        return self.params["dist"], self.params["iter"], self.params["vox"], self.params["eps"], self.params["samp"]

    def check(self):
        """
        Check if the GUI ha an update.
        """
        return self.hasUpdate

if __name__ == "__main__":
    try:
        PointCloudObstacleDetector()
    except rospy.ROSInterruptException:
        pass
