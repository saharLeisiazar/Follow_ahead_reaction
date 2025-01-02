# Overview
In human-robot interaction, especially in human-following applications, the robot must navigate in shared environments while consistently tracking a specific person. To maintain a clear view of the individual, a camera is mounted on the robot, paired with a motor to enable rotation. This setup ensures the camera can follow the target effectively.

This repository contains code to control the ZED2 camera's rotation using a Dynamixel servo motor. The motor adjusts the camera's orientation to keep the tracked person centered in the image.

# Requirements

1. Human Detection: The code subscribes to a topic to receive the detected human's location in the image.

2. Robot Odometry: If needed, the code subscribes to the robot's odometry to calculate and publish the human's pose in the global frame.