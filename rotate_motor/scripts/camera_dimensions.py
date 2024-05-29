import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class CameraDimensions:
    def __init__(self):
        rospy.init_node('camera_dim', anonymous=True)
        self.image_width = None
        self.image_height = None
        self.bridge = CvBridge()

        # Subscribe to the image topic to get camera dimensions once
        rospy.Subscriber("/zed2/rgb/image_raw", Image, self.image_callback, queue_size=1)

    def image_callback(self, msg):
        # Retrieve image dimensions from the first received message
        if self.image_width is None and self.image_height is None:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            self.image_height, self.image_width, _ = image.shape

            # Unsubscribe after getting dimensions
            rospy.Subscriber("/zed2/rgb/image_raw", Image, self.image_callback, queue_size=1)

def get_camera_dimensions():
    camera_dimensions = CameraDimensions()
    rospy.spin()
    return camera_dimensions.image_width, camera_dimensions.image_height
