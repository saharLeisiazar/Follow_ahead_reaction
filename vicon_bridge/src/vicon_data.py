import rospy
from geometry_msgs.msg import PoseStamped

def vicon_callback(data):
    # Process Vicon data
    print("Received Vicon data:", data)

def main():
    rospy.init_node('vicon_listener', anonymous=True)
    rospy.Subscriber('/vicon/object_name', PoseStamped, vicon_callback)
    rospy.spin()

if __name__ == '__main__':
    main()

