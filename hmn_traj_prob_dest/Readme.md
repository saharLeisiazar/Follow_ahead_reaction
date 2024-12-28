Human Action Prediction with LSTM-fc

This repository contains the implementation and resources for predicting the probabilities of a human's next possible action using a trained LSTM-fc model. The approach focuses on improving decision-making processes by incorporating human trajectory predictions.

Overview

Incorporating the probability of a human's next possible action enhances the performance of the decision-making process. This work addresses the challenges of predicting individual human actions by training a Long Short-Term Memory (LSTM) model with a fully connected (fc) layer to generate accurate probabilities for a human's next directional move: walking straight, turning right, or turning left.

Methodology

Initial Methods

Several recent off-the-shelf methods for predicting multiple human trajectories were tested (e.g., [1, 2, 3]). However, these methods did not yield robust and accurate results for individual human predictions. This limitation may be attributed to the training of these models on datasets (e.g., [4, 5]) featuring multiple individuals, where interactions influence individual behaviors.

Proposed Approach

To address this limitation, we trained an LSTM-fc model specifically designed to:

Sample a human's position over a three-second interval.

Generate probabilities for the next possible actions: walking straight, turning right, or turning left.

The fully connected layer enables the model to output the likelihood of each action.

Dataset

The Human3.6M dataset [6] was employed for training. The dataset, which includes various human motions, was filtered to focus exclusively on "walking" to align with the application's requirements. Key preprocessing steps included:

Downsampling: The dataset’s original frequency of 50 Hz was reduced to 5 Hz, resulting in approximately 700,000 points representing 2D positions of a walking person.

Sequence Extraction: Sequences of 15 consecutive points (three seconds at 5 Hz) were extracted. Each sequence's final point was designated as the ground truth, with preceding points serving as input for the model.

Training and Performance

The LSTM-fc model was trained to predict the probability distribution for the human’s future direction.

The model achieved an evaluation accuracy of 92.47% in predicting directional changes.

Repository Structure

data/: Contains preprocessed Human3.6M dataset samples (or scripts to preprocess the dataset).

models/: Includes the implementation of the LSTM-fc model.

training/: Scripts for training and evaluating the model.

notebooks/: Jupyter notebooks for exploratory data analysis and visualization.

README.md: This file.

Usage

Prerequisites

Python 3.8+

Required libraries listed in requirements.txt

Steps

Clone this repository:

git clone https://github.com/yourusername/human-action-prediction.git
cd human-action-prediction

Install dependencies:

pip install -r requirements.txt

Preprocess the dataset (if not already processed):

python preprocess_data.py

Train the model:

python train_model.py

Evaluate the model:

python evaluate_model.py

Citation

If you use this code or approach in your research, please cite the following references:

[1] Reference for off-the-shelf methods.

[6] Human3.6M dataset.

License

This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments

Special thanks to the creators of the Human3.6M dataset and the referenced trajectory prediction methods that inspired this work.

