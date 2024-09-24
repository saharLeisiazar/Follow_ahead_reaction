import rospy
from nav_msgs.msg import OccupancyGrid
import numpy as np


class node():
    def __init__(self):
        rospy.init_node('main', anonymous=True)
        rospy.Subscriber("/map", OccupancyGrid, self.map_callback, buff_size=1)
        self.pub = rospy.Publisher("/new_map", OccupancyGrid, queue_size=1)


    def map_callback(self, data):
        res = data.info.resolution
        origin_x = data.info.origin.position.x
        origin_y = data.info.origin.position.y
        width = data.info.width
        original_data = data.data

        m = OccupancyGrid()
        m.info = data.info
        m.header.frame_id = 'map'
        m.header.stamp = rospy.Time.now()


        x_min = 3.45
        x_max = 3.6
        y_min = -3.
        y_max = -1.6
        idx_list = []
        for new_y in np.arange(y_min, y_max, res):
            for new_x in np.arange(x_min, x_max, res):

                x_idx = int(np.rint((new_x - origin_x) / res))
                y_idx = int(np.rint((new_y - origin_y) / res))
                idx = int(x_idx + width * y_idx)

                p=0.9
                if np.random.choice(2, p=[1-p, p]):
                    idx_list.append(idx)



        new_data =[]
        for i in range(len(data.data)):
            if i in idx_list :  #and original_data[i]
                new_data.append(100)
            else:
                new_data.append(original_data[i])
                # new_data.append(0)

        m.data = new_data
        # m.data = []
        self.pub.publish(m)

        return


if __name__ == '__main__':
    node()
    rospy.spin()        