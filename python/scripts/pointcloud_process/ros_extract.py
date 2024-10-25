import rospy
from sensor_msgs.msg import PointCloud2, Image
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
import os
from utils.filesystem import recursive_mkdir

FILE_ROOT = "/mnt/e/Workspace/CathederTelesurgery/Data/RosDump/2024-0924"
recursive_mkdir(f"{FILE_ROOT}/Depth")
recursive_mkdir(f"{FILE_ROOT}/RGB")


__callcount__ = [0,0]
DEPTH_TOPIC = "/device_0/sensor_0/Depth_0/image/data"
RGB_TOPIC  = "/device_0/sensor_1/Color_0/image/data"

def pointcloud_callback(data):
    print(__callcount__)
    shape = (data.height, data.width)
    pc_data = np.frombuffer(data.data, dtype=np.uint16)
    pc_data = pc_data.reshape(shape)    
    np.save(f'{FILE_ROOT}/Depth/{__callcount__[0]}.npy', pc_data)
    __callcount__[0]+=1
def rgb_callback(data):
    print(__callcount__)
    bridge = CvBridge()
    try:
        cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
        cv2.imwrite(f'{FILE_ROOT}/RGB/{__callcount__[1]}.png', cv_image)
    except CvBridgeError as e:
        print(e)
    __callcount__[1]+=1
def main():
    rospy.init_node('data_exporter', anonymous=True)
    rospy.Subscriber(DEPTH_TOPIC, Image, pointcloud_callback)
    # rospy.Subscriber(RGB_TOPIC, Image, rgb_callback)
    rospy.spin()
if __name__ == '__main__':
    __callcount__ = [0,0]
    main()