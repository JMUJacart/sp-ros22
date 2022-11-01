#!/usr/bin/env python

"""
@author  Matthew Dim and Jacob Bringham :)
@version 4/26/22 
"""


import rospy
import math
import numpy as np
import time
from std_msgs.msg import Float64
  
from navigation_msgs.msg import LatLongPoint

class SpeedEstimator(object):
    
    def __init__(self):
        
        self.name        = rospy.get_param("name", "gps_speed")
        # Time interval to calculate velocity over
        self.interval = rospy.get_param("interval", 1.0)

        # GPS calculated speed topic miles/h
        rospy.init_node(self.name)
        self.last_msg = None
        self.vel_mps = 0 # Meters / Seconds
        self.gps_data = rospy.Subscriber('/gps_send', LatLongPoint, self.GPSVelocity)
        self.gps_vel_pub = rospy.Publisher('/gps_estimated_velocity', Float64, queue_size = 10)
        
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            rate.sleep()
 
    def GPSVelocity(self, msg):

        # Fix bad messages
        if msg.header.stamp.to_sec() == 0:
            msg.header.stamp = rospy.Time.now()
        
        # Need a prior message to perform calculation
        if self.last_msg == None:
            self.last_msg = msg
            return

        # First track the current time of msg
        delta = msg.header.stamp.to_sec() - self.last_msg.header.stamp.to_sec()

        if delta >= self.interval:
            # Call distance helper and convert to m/s
            dist = self.GPSDistance(msg.latitude, msg.longitude, self.last_msg.latitude, self.last_msg.longitude)
            self.vel_mps = dist / delta
            
            # Publish the velocity!
            vel = Float64()
            vel.data = self.vel_mps
            self.gps_vel_pub.publish(vel)

            self.last_msg = msg
            
        
    def GPSDistance(self, lat1, long1, lat2, long2):
        
        # Convert from degrees to radians
        lat1  = lat1  * math.pi / 180
        long1 = long1 * math.pi / 180
        lat2  = lat2  * math.pi / 180
        long2 = long2 * math.pi / 180
        
        # Radius of earth in meters
        r = 6378100 

        point1 = r * math.cos(lat1)
        x1 = point1 * math.cos(long1)
        y1 = point1 * math.sin(long1)
        z1 = r * math.sin(lat1)

        point2 = r * math.cos(lat2)
        x2 = point2 * math.cos(long2)
        y2 = point2 * math.sin(long2)
        z2 = r * math.sin(lat2)

        a = np.array([x1, y1, z1])
        b = np.array([x2, y2, z2])

        return np.linalg.norm( a - b )


if __name__ == "__main__":
    try:
        SpeedEstimator()
    except rospy.ROSInterruptException:
        pass
