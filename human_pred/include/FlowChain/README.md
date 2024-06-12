# FlowChain
(June 3-11, Week 4)

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




