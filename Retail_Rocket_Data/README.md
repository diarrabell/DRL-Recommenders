The following sections discuss using the Retail Rocket dataset with the modified SNQN recommender that incorporates contextual item features.

## Setting Up
If you need an API token for downloading datasets from Kaggle, please follow [the directions](https://github.com/Kaggle/kaggle-api) under "API credentials".

## Train and Evaluate using Google Colab
1. Download `RR_SNQN_Recommender.ipynb`.
2. Go to [Google Colab](https://colab.research.google.com/) and upload `RR_SNQN_Recommender.ipynb` (by selecting "File", then "Upload notebook").
3. Select GPU as the runtime: Click on "Runtime", then "Change runtime type", and choose "GPU" under "Hardware accelerator", then hit "Save".
4. Run all the cells in the notebook:
    - The first cell clones the repository.
    - The second cell installs needed libraries.
    - The third cell will prompt you to upload the `kaggle.json` token.
    - The fourth cell processes the data for training. All files are in the src folder, and the bolded files are from the SA2C source code.
        - Download the Retail Rocket zip file from Kaggle (`download.sh`)
        - Create sessions from events.csv and split into training, validation, and test sessions (**`preprocess_and_split.py`**)
        - Create item features from item_properties_part1.csv and item_properties_part1.csv (`item_features.py`)
        - Create the replay buffer (**`replay_buffer.py`**)
    - The fifth cell trains the model and evaluates it on the validation sessions, then evaluates it on the test sessions. (The original files for training have been upgraded to TensorFlow 2 and have been renamed to end with "new".)

## Data Processing
The following summarizes `main_setup.py`, which contains the steps for processing the input files for training.

The two item properties CSV files are combined. Using the events CSV file, filter out "transaction" events, as well as users and items that had fewer than three interactions. Each item in the events should have corresponding information on it for training, so interactions with item IDs that aren't found in item properties will also be filtered out. Label encode the users and items, and add a column to indicate whether the interaction is a click (0 for "view") or purchase (1 for "addtocart"). To create user sessions, sort the interactions by user, then timestamp.

Shuffle the session IDs, then split the sessions into training, validation, and test sessions, with the split being 80/10/10. 

Item features are created using the combined properties files. The properties are filtered for item IDs that appeared in the sessions, and the item label encoder from above is used to encode the item IDs for the properties. To approximate the item state for each interaction, filter for the item property with the latest timestamp. Since the model takes in floats, the strings from the "value" column are converted: They are encoded and hashed, and the outputs are converted to hexadecimal strings, then converted to decimals.

A replay buffer is created from the training sessions. The current state consists of the the last ten items interacted with (with padding added if needed), the action is the item that was interacted with given the current state, and the next state consists of the most recent ten interactions including the action. Since there are two types of interactions, whether the action is a purchase or not is also recorded. There is a 'is_done' boolean indicator that will be set to true once all sequences have been recorded for the session.

## Incorporating Item Features into the SNQN Recommender
There are a few modifications to the SNQN file, `SNQN_new.py`, to incorporate contextual item features:
- A new parameter, the lambda value, which is used a mixing parameter in equation 2 of the [HRNN paper](https://assets.amazon.science/96/71/d1f25754497681133c7aa2b7eb05/temporal-contextual-recommendation-in-real-time.pdf), can be passed in as input.
- The input item features are encoded by adding another fully connected layer.
- The values for phi prime and phi tilde from equation 2 of the HRNN paper are computed, and phi tilde is specified as the logits for calculating the cross entropy loss.

## Training the GRU-SNQN Model
Below are some of the settings used for modelling. All settings except for the number of epochs and the lambda value were default settings from the original source code.
- batch size: 256
- embedding size: 64
- learning rate: 0.005 
- discount factor: 0.5
- optimizer: Adam
- number of epochs: 15
- lambda value for incorporating item features: 0.2

While the training sessions are used for training the model, the validation sessions are used to evaluate the model every 9000 steps, or batches. The test sessions are used to evaluate the model after training finishes.

## Results for Clicks
#### Hit Ratio at K
|  HR@5 | HR@10 | HR@15 | HR@20 |
| --- | --- | --- | --- |
| 0.1710 | 0.2124 | 0.2361 | 0.2526 |

####  Normalized Discounted Cumulative Gain at K
|  NG@5 | NG@10 | NG@15 | NG@20 |
| --- | --- | --- | --- |
| 0.1300 | 0.1434 | 0.1496 | 0.1535 |

## Results for Purchases
#### Hit Ratio at K
|  HR@5 | HR@10 | HR@15 | HR@20 |
| --- | --- | --- | --- |
| 0.2931 | 0.3382 | 0.3660 | 0.3861 |

####  Normalized Discounted Cumulative Gain at K
|  NG@5 | NG@10 | NG@15 | NG@20 |
| --- | --- | --- | --- |
| 0.2330 | 0.2477 | 0.2550 | 0.2600 |

## Links to Resources Used
- Paper and original source code: ["Supervised Advantage Actor-Critic for
Recommender Systems"](https://arxiv.org/pdf/2111.03474.pdf)
- Paper: ["Temporal-Contextual Recommendation in Real-Time"](https://assets.amazon.science/96/71/d1f25754497681133c7aa2b7eb05/temporal-contextual-recommendation-in-real-time.pdf)
- [Retail Rocket dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)
