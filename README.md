# Adapting to Frequent Human Direction Changes in Autonomous Frontal Following Robots

This repository implements the approach proposed in my paper [1] to address the challenges of robot follow-ahead applications, where human behavior can be highly variable and unpredictable. Our method leverages a decision-making technique to improve the robot's ability to navigate effectively in front of a person.

## Overview
A novel methodology is developed comprising three integrated modules: RL, LSTM, and MCTS. This approach introduces a unique consideration of distinct action spaces for humans and robots, enabling the system to dynamically capture and adapt to sudden changes in human trajectories with reasonable probability. This integration builds on prior work [2] by improving responsiveness and adaptability, addressing challenges in scenarios with frequent and unpredictable human direction changes. 
![alt text](images/cover.png)


## Key Modules

### Tree Expansion:
Monte Carlo Tree Search (MCTS) is used to determine optimal actions for a robot following a person ahead. This approach enhances traditional MCTS by integrating a trained reinforcement learning (RL) model for node evaluation and an LSTM-based model to predict human action probabilities. By combining these elements, the method improves decision-making accuracy and robustness, enabling the robot to effectively balance exploration and exploitation in complex, dynamic scenarios.

At each time step, the system expands a tree with two layers: robot nodes and human nodes. 
Robot nodes represent possible actions or states of the robot.
Human nodes represent potential states or movements of the person being followed.
While expanding the tree and creating new leaf nodes, the safety of each node is evaluated. If a node directs the robot toward an unsafe region, it is removed from the tree.
The RL model evaluates each node to estimate its value.
The Human Trajcetory Prediction model assigns probabilitis to human nodes.

The expansion process is done by MCTS at a frequency of 5 Hz.
After expanding the tree, the child node with the highest visit count and expansion is chosen as the optimal action for the robot.
A new tree is expanded based on the updated state and inputs.


### Reinforcement Learning (RL) Model:

Evaluates the quality of tree nodes during the tree expansion.

### Human Trajcetory Prediction Model:
An LSTM-fc model is trained specifically to sample a human’s position over a three-second interval and generate probabilities for their next possible actions.
The fully connected layer attached to the LSTM enables the model to output the likelihood of the human walking straight, turning right, or turning left. For training, 
the Human3.6M dataset is employed [3].



## Citation
[1] Leisiazar S, Razavi R, Park EJ, Lim A, Chen M. Adapting to frequent human direction changes in autonomous frontal following robots. 

[2] Leisiazar S, Park EJ, Lim A, Chen M. An MCTS-DRL Based Obstacle and Occlusion Avoidance Methodology in Robotic Follow-Ahead Applications. In2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2023 Oct 1 (pp. 221-228). IEEE.

[3] Catalin Ionescu, Dragos Papava, Vlad Olaru, and Cristian Sminchisescu. Human3.6m: Large scale datasets and predictive methods for 3d human sensing in natural environments. IEEE transactions on pattern analysis and machine intelligence,36(7):1325–1339, 2013.


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
