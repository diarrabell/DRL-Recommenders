# Deep Reinforcement Learning Recommenders

AIPI 531 Take Home Project

Diarra Bell and Stephanie Horng

## Project Objectives
- Train variations of the SNQN ([Supervised Negative
Q-learning](https://arxiv.org/pdf/2111.03474.pdf])) product recommendation recommender for two E-commerce use cases and compare the performances.
- Include item features as side information for cold-start items.
- Use two offline evaluation metrics for benchmarking.

## Datasets Overview
The following datasets were chosen for training.
#### H&M Dataset
H&M is a global commerce company that specializes in Women's and Men's fashion. The dataset we are using comes from a Kaggle competition where users are tasked to use past H&M data to provide product recommendations: https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data 
The dataset files we are using for this project are: 
*transactions_train.csv* - consisting of previous purchases made by customers. Information includes item ID, customer ID, and price, and date.
*articles.csv* - consists of item characteristics such as appearance, color and category.
#### Retail Rocket Dataset
Retail Rocket provides shoppers with personalized real-time recommendations through multiple channels. The [dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) contains four files that includes raw data collected from a real-world ecommerce website. All files except for category_tree.csv are used. events.csv contains interaction information for each user, including the timestamp of the interaction, the event type (click, add to cart, or transaction), and the item ID. There are two item_properties.csv files that provide more details on the items for about 90% of the interactions from events.csv. The item property and property values have been hashed for confidentiality purposes.  

## SNQN Recommender Overview
The SNQN, or Supervised Negative Q-learning, recommender, addresses the challenges of applying reinforcment learning algorithms directly. SNQN addresses the issue with no real-time feedback during off-policy learning by combining supervised sequential learning with reinforcement learning. Additionally, there often is a lack of sufficient reward signals available, especially negative reward signals, and using only positive reward signals causes the Q-values to be overestimated. The Q-learning loss function of SNQN incorporates action rewards for a sampled set of negative, or unobserved, actions in addition to the positive ones. The deep reinforcement learning portion therefore acts as a self-regularizer by giving higher rewards to positive actions.

In this project, GRU is used as the base model, and it is used twice per dataset with the SNQN recommender: The original SNQN recommender will be used, along with a modified SNQN recommender that incorporates item features.

## Incorporating Item Features into the SNQN Recommender
The project uses some components from the **HRNN-meta**, or hierarchical recurrent network with meta data, model. HRNN-meta incorporates user and item features into the recommendation systems through learned embeddings in order to adapt to preferences in real-time and address the "cold-start" problem for new users and items that have not been interacted with yet. 

More specifically, the project uses equation 2 of the [HRNN paper](https://assets.amazon.science/96/71/d1f25754497681133c7aa2b7eb05/temporal-contextual-recommendation-in-real-time.pdf) to incorporate item features only. The learned weights and bias of the item feature vectors are used to calculate phi-prime, and phi-prime and the logits from the GRU model are combined using a lambda value to get the phi-tilde value, which is used as the logits for calculating the cross-entropy loss. 

## Steps for Setting Up and Running the Code
For H&M, open the H&M_Recommenders.ipynb notebook in the H_and_M_Data folder and run all cells. 
To run this notebook, you will need to have access to the [Kaggle API](https://github.com/Kaggle/kaggle-api).  When prompted, upload your kaggle.json file to continue running the file.
The first model trained is an SNQN model without item features.
The second model trained is a GRU4Rec model with item features.

For Retail Rocket, please refer to first two sections from the [README for the dataset](https://github.com/sfhorng/AIPI-531-Final-Project/tree/main/Retail_Rocket_Data).

## Evaluation Metrics and Results
The offline evaluation metrics used are Hit Ratio (HR@k) and Normalized Discounted Cumulative Gain (NDCG@k). HR@k is the ratio of ground-truth items that are in the top k slots of the recommendation ranking. NDCG@k represents how relevant the top k ranked items are, and a higher score indicates that relevant products are given higher rankings.

The following results are for purchases.
### Results for the H&M Dataset Without Item Features
#### Hit Ratio at K
|  HR@5 | HR@10 | HR@15 | HR@20 |
| ---   |  ---  |  ---  |  ---  |
| 0.019 | 0.026 | 0.033 | 0.036 |

####  Normalized Discounted Cumulative Gain at K
|  NG@5 | NG@10 | NG@15 | NG@20 |
| ---   | ---   | ---   | ---   |
| 0.014 | 0.016 | 0.018 | 0.019 |

### Results for the H&M Dataset With Item Features
#### Hit Ratio at K
|  HR@5 | HR@10 | HR@15 | HR@20 |
| --- | --- | --- | --- |
|     |     |     |     |

####  Normalized Discounted Cumulative Gain at K
|  NG@5 | NG@10 | NG@15 | NG@20 |
| --- | --- | --- | --- |
|     |     |     |     |

### Results for the Retail Rocket Dataset Without Item Features
#### Hit Ratio at K
|  HR@5 | HR@10 | HR@15 | HR@20 |
| --- | --- | --- | --- |
|  0.4410 | 0.5007 | 0.5294 | 0.5471 |

####  Normalized Discounted Cumulative Gain at K
|  NG@5 | NG@10 | NG@15 | NG@20 |
| --- | --- | --- | --- |
| 0.3579| 0.3774| 0.3850| 0.3892 |

### Results for the Retail Rocket Dataset With Item Features
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
- [H&M dataset](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data)
- [Retail Rocket dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)
- Article on NDCG: ["Understanding NDCG as a Metric for your Recommendation System"](https://medium.com/@readsumant/understanding-ndcg-as-a-metric-for-your-recomendation-system-5cd012fb3397#:~:text=Normalized%20Discounted%20Cumulative%20Gain%20or,relevant%20products%20are%20ranked%20higher.)
- [AIPI student repository](https://github.com/architkaila/recommenders_aipi590)
