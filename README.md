# TUTR

# Data Preprocessing and TUTR Model Re-training (May 13-24, Week 1 & 2)

## Literature Review
- Reviewed academic paper related to the TUTR model to understand the foundational concepts and methodologies.
- Explored the GitHub repositories of TUTR project to understand the implementations.

## Dataset Preparation
- Downloaded and analyzed the dataset used by the original TUTR model to familiarize with the data structure and requirements.
- Obtained the SFU_nav_store dataset from a collaborator, Zhitian, which was used to re-train the TUTR model.

## Data Transformation
- Utilized `transform_csv.py` to convert the SFU_nav_store dataset into a format compatible with the TUTR model's input requirements.
    ```bash
    python transform_csv.py
    ```
    - Data reading
        - Load the raw dataset (put inside folder `dataset`). We only focus on the files called 'vicon_hat_3_hat_3_translation.csv' in each of the folders named by date, as this is the dataset that includes the corresponding position data.
        - These files contain the following fields: timestamp, x, y, z.
    - Data Interpolation and Resampling
        - The 'Time' column is converted from a UNIX timestamp to a datetime object. 
        - Set 'Time' as the DataFrame index and performs resampling at a consistent interval of 400 milliseconds. 
        - After resampling, 'Time' is reset as a column and then transformed into 'frame_id' representing milliseconds since the start of the dataset. 
    - Data Transformation
        - agent_id is the id for each agent recorded. Therefore each trajectory file is assigned a unique id.
        - x and y positions are directly use of the position data
        - group is set to be default as 'human'
    - File output
        - the output datafile is in the format of: 
        ```bash
        frame_ID:int  agent_ID:int  pos_x:float  pos_y:float  group:str
        ```
        - Since there are 145 ouput data files and each contains a single trajectory of an agent, the train and test split is done by slecting 29 of the datafiles and put into the test folder. 
        
- Ran `get_data_pkl.py` to transform the processed CSV files into pickle (.pkl) files, which are utilized for model training.
    ```bash
    python get_data_pkl.py --train data/sfu/train --test data/sfu/test --config config/sfu.py
    ```
    `config/sfu.py` file contains configuration settings for model training. 

## Model Training and Testing
- Executed `train.py` using the newly created .pkl file to train the TUTR model. 
    ```bash
    python train.py --dataset_name sfu --hp_config config/sfu.py --gpu 0
    ```
- The best performing model weights were saved as `best.pth`, which were later used for testing the model's efficacy.


## (May 27 - 31, Week 3)
## Trained Model Testing

- Implemented the testing function for TUTR in the file `script/test.py` to visualize the prediction with probabiity of next step beased on last 8 observed positions. 
- The trained model is saved in the file `best_sfu_2.pth`.
- Results are saved to the same output as the vae model in the `fig` folder.


# FlowChain
## (June 3-11, Week 4)

## Literature Review
- Reviewed academic paper related to the FlowChain-ICVV2023 model to understand the foundational concepts and methodologies.
- [Fast Inference and Update of Probabilistic Density Estimation on Trajectory Prediction](https://arxiv.org/abs/2308.08824) by Takahiro Maeda and Norimichi Ukita.
- Explored the GitHub repositories of [FlowChain-ICVV2023](https://github.com/meaten/FlowChain-ICCV2023) project to understand the implementations.


## Dataset Preparation
- Modified the testing dataset for 'eth' - agent2 to be a U-shape trjectory.
- Process the modified dataset to get new processed_data and used for testing.

```
python src/data/TP/process_data.py
```

## Direction Prediction and Visualization
- created new functions to make predictions about the next step direction (straight, right, left).
- Added visualization functions to plot a graph regarding the direction predictions.
- The added functions can be found in TP_visualizer.py
- Pretrained models can be found [here](https://drive.google.com/drive/folders/1bA0ut-qrgtr8rV5odUEKk25w9I__HjCY?usp=share_link)

- Just download the 'output' folder to the root of this repo, and it's ready to test these models (already downloaded).


## Testing
without visualization
```
python src/main.py --config_file config/TP/FlowChain/CIF_separate_cond_v_trajectron/eth.yml --mode test
```

with visualization
```
python src/main.py --config_file config/TP/FlowChain/CIF_separate_cond_v_trajectron/eth.yml --mode test --visualize
```

## Model Training
For example of ETH split,
```
python src/main.py --config_file config/TP/FlowChain/CIF_separate_cond_v_trajectron/eth.yml --mode train
```

# Sensor detecting and tracking Appication
# (June 19 - July 3, Week 6-7)

- Studied the camera and motor documentation to begin implementing the application.
- Implemented the application to rotate the motor when the camera detects a human and to keep rotating (tracking) with the human to keep them centered in the camera’s view.
- Tested the performance of the sensor rotation system. The system performs well under normal conditions but has some exceptions:
    1. When the human is too close to the camera, their face is not visible to the camera, and the system starts searching for a human.
    2. When the human is moving too fast, the camera might lose tracking and start detecting again.
    3. When several humans appear in the camera view (including the robot in the lab, which is also recognized as human), the system stops tracking the non-moving human. It may be necessary to cover the robots during testing.

```
After correctly connecting all the sensors, please run the track_human.py file from VScode directly, the rosrun would run another file.
```

# Converting human local coordinates to global coordinates and publish
# (July 5 - July 12, Week 8)

- Converted the human’s position from the local coordinate system detected by the camera to the global coordinate system of the robot and published the human’s position in the global system.
- Implemented the testing of human positions with plots, generating a plot after each session terminates which will help visulize the human trejectory.
- There is still an unfinished part: I encountered an issue with the motor angle. The current code uses the relative degree of the motor, which causes incorrect position calculations. I attempted to change this to the absolute position of the motor using a direct service call, but for some reason, the service call isn’t working. I haven’t figured out the cause yet, so I commented out this part of the code (get_current_motor_angle function).

```
After correctly connecting all the sensors and the robot, please run the track_human.py file from VScode directly, the rosrun would run another file.
```

# Implementing PID and PWM conroller to improve performance of sensor
# (August 6 - August 15, Week 12)

Developed and integrated both a Proportional-Integral-Derivative (PID) controller and a Pulse Width Modulation (PWM) controller to manage the angle adjustment of a camera in response to human target detection. The goal was to keep the human target centered in the camera’s field of view, even as they move around.


## Overview

The code is designed to dynamically adjust the camera's orientation based on the position of a detected human target. It provides two primary methods for controlling the camera's angle:

- **PID Controller**: The PID controller continuously adjusts the camera angle by minimizing the error between the target’s position in the frame and the center of the frame. It balances responsiveness and stability by combining proportional, integral, and derivative components. The PID controller is particularly effective for smooth, precise adjustments.
  - **Key Code References**:
    - PID Calculation: `pid_controller(self, error)`
    - Angle Adjustment: `pid_angle_calculation(self, mean_bb, image_center)`

- **PWM Controller**: The PWM controller adjusts the camera’s angle by varying the duty cycle based on the angular error. Larger errors result in a higher duty cycle, leading to faster corrections. A deadband and gradual duty cycle adjustments help reduce oscillations, making the PWM controller suitable for responsive but controlled movements.
  - **Key Code References**:
    - PWM Duty Cycle Calculation: `pwm_controller(self, angular_error)`
    - Angle Adjustment: `pwm_angle_calculation(self, mean_bb, image_center)`

## Usage Instructions

### Switching Between Controllers

You can easily switch between the PID controller, PWM controller, and the original angle adjustment method by setting the following flags in the `__init__` method:

- **Enable PID Controller**:
  ```python
  self.use_pid = True
  self.use_pwm = False  

- **Enable PWM Controller**:
  ```python
  self.use_pid = False
  self.use_pwm = True  

- **Enable Original method**:
  ```python
  self.use_pid = False
  self.use_pwm = False  

# Continued on Global Position Conversion Adjustments
# (August 16 - August 23, Week 14)

### Overview

I refined the conversion of the detected human position from local (camera-relative) coordinates to global coordinates. The original implementation did not correctly account for the robot's orientation and the absolute position of the motor, leading to inaccuracies in the global position coordinates. My updates ensure that these factors are properly integrated into the calculation, resulting in more accurate global positioning.

### Enhancements

The following improvements were made to the global position conversion process:

- **Incorporation of Robot Orientation**: The robot's current orientation is now factored into the conversion process. This ensures that the global coordinates reflect the true position of the human relative to the robot's heading.
- **Motor Position Integration**: The absolute position of the motor is also included in the calculations. This adjustment accounts for the camera's rotation and correctly maps the detected position to the global frame of reference.

### Key Code References

- **Global Position Calculation**: The core logic for converting local coordinates to global coordinates can be found in the `transform_to_global(self, local_coords)` method.
- **Robot Orientation Handling**: The robot's orientation is retrieved and used within the `robot_position_callback(self, data)` method.
- **Motor Position Handling**: The motor's absolute position is captured and applied in the `state_callback(self, data)` and `get_motor_position(self)` methods.

### Usage Instructions

The enhanced global position conversion is integrated directly into the main workflow of the camera control system. When the system detects a human target, the updated coordinates are automatically calculated and published.

- **Conversion Process**: The global position is calculated each time a human target is detected and updated in the `transform_to_global(self, local_coords)` method.
- **Ensure Accuracy**: To maintain accurate positioning, ensure that the robot's orientation and motor position are correctly updated by regularly calling `robot_position_callback(self, data)` and `state_callback(self, data)`.

