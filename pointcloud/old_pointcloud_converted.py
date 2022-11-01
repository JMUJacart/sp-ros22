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
    -1: {"r": 0.5, "g": 0.5, "b": 0.5},
     0: {"r": 0.5, "g": 0,   "b": 0.75},
     1: {"r": 75,  "g": 0,   "b": 130},
     2: {"r": 0,   "g": 0,   "b": 255},
     3: {"r": 255, "g": 255, "b": 0},
     4: {"r": 255, "g": 127, "b": 211},
}

class PointCloudObstacleDetector(object):
    
    def __init__(self):
        
        # ----- Parameters -----
        self.name     = rospy.get_param("name", "converter")
        # self.cloud_in = rospy.get_param("cloud_in", "/front_cam/front/point_cloud/cloud_registered")
        self.cloud_in   = rospy.get_param("cloud_in", "/voxel_grid/output")
        self.cloud_out  = rospy.get_param("cloud_out", "/object_cloud")
        self.ransac_out = rospy.get_param("ransac_out", "/ransac_cloud")

        rospy.init_node(self.name)
        
        # ----- Node State -----
        self.cloud_sub  = rospy.Subscriber(self.cloud_in,  PointCloud2, self.findClusters)
        self.cloud_pub  = rospy.Publisher(self.cloud_out,  PointCloud2, queue_size=10)
        self.ransac_pub = rospy.Publisher(self.ransac_out, PointCloud2, queue_size=10)

        r = rospy.Rate(1)
        while not rospy.is_shutdown():
            r.sleep()

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

        Default ZED2i behaviour sends a pointcloud at around 10Hz, so this needs to be performant.

        @param msg  Incoming PointCloud2 message
        """

        total = 0

        s = time.time()

        # Convert the PointCloud into a 2D numpy array of (x, y, z, rgba)
        data = np.fromiter(pc2.read_points(cloud=msg, field_names=None, skip_nans=True, uvs=[]), \
            dtype="f4,f4,f4,f4").view(np.float32).reshape(-1, 4)
        
        # Strip the color info
        data = np.delete(data, 3, axis=1)

        f = time.time() - s
        total += f
        print("readPoints(N=%d) took %f" % (data.shape[0], f))


        s = time.time()
        # Locate the ground plane via RANSAC (the inliers)
        inliers, outliers = self.RANSAC(data, 0.3, 50)

        f = time.time() - s
        total += f
        print("RANSAC(N=%d) took %f" % (data.shape[0], f))

        print("Input: %s, Output: %s, %s" % (str(data.shape), str(inliers.shape), str(outliers.shape)))
        
        s = time.time()
        # Publish a PointCloud of the inliers and outliers for debugging
        self.toPointCloud2(inliers, outliers, msg.header)

        f = time.time() - s
        total += f
        print("toPointCloud2() took %f" % (f))


        s = time.time()
        # Cluster the points using DBSCAN
        model = DBSCAN(eps=2.5, min_samples=2)
        pred = model.fit_predict(outliers)
        f = time.time() - s
        total += f
        print("DBSCAN(N=%d) took %f" % (outliers.shape[0], f))

        print(np.asarray(np.unique(pred, return_counts=True)).T)

        print("Cloud #%d took %f sec" % (msg.header.seq, total))
        
        # Publish a PointCloud relating points to obstacles for debugging
        r = np.array([])
        g = np.array([])
        b = np.array([])
        for i in range(pred.shape[0]):
            o = ObjectColorMap[pred[i]]
            r = np.append(r, o["r"])
            g = np.append(g, o["g"])
            b = np.append(b, o["b"])
        
        r = r.reshape(r.shape[0], 1)
        b = r.reshape(b.shape[0], 1)
        g = r.reshape(g.shape[0], 1)

        # print(outliers.shape)
        # print(r.shape)
        # print(g.shape)
        # print(b.shape)

        merged = np.hstack((outliers, r, g, b, np.ones(outliers.shape[0]).reshape(outliers.shape[0], 1)))
        print(merged[0])

        cloud_out = self.point_cloud(merged, msg.header)
        self.cloud_pub.publish(cloud_out)
        
               


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
        print(merged.dtype)
        merged = merged.astype(np.float32)
        print(merged[0])

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


if __name__ == "__main__":
    try:
        PointCloudObstacleDetector()
    except rospy.ROSInterruptException:
        pass
