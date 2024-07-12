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




