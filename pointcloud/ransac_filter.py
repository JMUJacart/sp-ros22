#!/usr/bin/env python

"""

rosrun tf static_transform_publisher 0 0 0 0 0 0 map front_cam_camera_center 100
"""

import rospy
import numpy as np
import math
import time
import tf

# Messages
from navigation_msgs.msg import VehicleState, Obstacle, ObstacleArray
from geometry_msgs.msg import TwistStamped, Vector3, PointStamped
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2
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

class PointCloud2Converter(object):
    
    def __init__(self):
        
        # ----- Parameters -----
        self.name     = rospy.get_param("name", "converter")
        # self.cloud_in = rospy.get_param("cloud_in", "/front_cam/front/point_cloud/cloud_registered")
        self.cloud_in = rospy.get_param("cloud_in", "/voxel_grid/output")

        rospy.init_node(self.name)
        
        self.cloud_sub = rospy.Subscriber(self.cloud_in, PointCloud2, self.writeCloud)

        # Repeatedly merge obstacles and publish results until shutdown
        r = rospy.Rate(1)
        while not rospy.is_shutdown():
            r.sleep()

    def printCloud(self, msg):
        """
        Prints out information about a PointCloud, not including the data.

        @param self
        @param msg
        """
        print("Width x Height: %dx%d" % (msg.width, msg.height))  

        for i, pf in enumerate(msg.fields):
            print("Name: %s\nOffset: %u\nDatatype: %s\nCount: %d\n" % (pf.name, pf.offset, PointFieldDataTypes[pf.datatype], pf.count))

        print("BigEndian: %s" % (str(msg.is_bigendian)))
        print("PointStep: %u" % (msg.point_step))
        print("RowStep: %u" % (msg.row_step))
        print("IsDense: %s" % (str(msg.is_dense)))


    def writeCloud(self, msg):
        """
        Writes a pointcloud to a file.

        The ZED sends these at about 10HZ

        @param self
        @param msg  PointCloud2
        """
        print("I received a PointCloud #%d from %s" % (msg.header.seq, self.cloud_in))

        s = time.time()
        points2 = np.fromiter(pc2.read_points(cloud=msg, field_names=None, skip_nans=True, uvs=[]), \
            dtype="f8,f8,f8,f8")
        print("readPoints() took %f" % (time.time() - s))

        print(points2)


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
    try:
        PointCloud2Converter()
    except rospy.ROSInterruptException:
        pass
