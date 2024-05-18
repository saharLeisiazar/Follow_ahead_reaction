import rospy
from nav_msgs.msg import OccupancyGrid
import math

class node():
    def __init__(self):
        rospy.init_node('crop', anonymous=True)

        rospy.Subscriber("/map", OccupancyGrid, self.callback, buff_size=1)
        self.pub_map = rospy.Publisher("/croped_map", OccupancyGrid, queue_size = 1)
        self.x_min = -10.
        self.x_max = 10.
        self.y_min = -10.
        self.y_max = 10.


    def callback(self, data):
        m = OccupancyGrid()
        m.header = data.header
        m.info = data.info


        new_data = []
        for i in range(len(data.data)):
            x = (i % data.info.width) * m.info.resolution + data.info.origin.position.x
            y = math.floor(i / data.info.width) * m.info.resolution + data.info.origin.position.y

            if x<=self.x_max and x>=self.x_min and y>=self.y_min and y<=self.y_max:
                new_data.append(data.data[i])

        m.data = new_data
        m.info.origin.position.x = self.x_min
        m.info.origin.position.y = self.y_min
        self.pub_map.publish(m)

if __name__ == '__main__':
    node()
    rospy.spin()        