#!/usr/bin/env python

"""
@author  Matthew Dim
@version 4/26/22 
"""


import rospy
from std_msgs.msg import Float32

class SpeedChange(object):
    
    def __init__(self):
        
        self.name        = rospy.get_param("name", "speed_change")
        rospy.init_node(self.name)
        self.ui_speed = rospy.Subscriber('/speed_setting', Float32, self.ChangeCartSpeed)
        self.change_vel = rospy.Publisher('/speed', Float32, queue_size = 1)
        
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            rate.sleep()
 
    def ChangeCartSpeed(self, msg): 
       rospy.loginfo("Cart speed changed: %f", msg.data)
       self.change_vel.publish(msg.data) 
if __name__ == "__main__":
    try:
        SpeedChange()
    except rospy.ROSInterruptException:
         pass
