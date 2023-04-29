# Deep Reinforcement Learning Recommenders

AIPI 531 Take Home Project

## Project Objectives
- Train the SNQN ([Supervised Negative
Q-learning](https://arxiv.org/pdf/2111.03474.pdf])) product recommendation recommender
for two E-commerce use cases.
- Include item features as side information for cold-start items.
- Use two offline evaluation metrics for benchmarking.

## Datasets Overview
The following datasets were chosen for training.
#### H&M Dataset

#### Retail Rocket Dataset
Retail Rocket provides shoppers with personalized real-time recommendations through multiple channels. The [dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) contains four files that includes raw data collected from a real-world ecommerce website. All files except for category_tree.csv are used. events.csv contains interaction information for each user, including the timestamp of the interaction, the event type (click, add to cart, or transaction), and the item ID. There are two item_properties.csv files that provide more details on the items for about 90% of the interactions from events.csv. The item property and property values have been hashed for confidentiality purposes.  

## GRU-SNQN Model Overview

## HRNN-meta Model Overview

## Steps for Setting Up and Running the Code
For H&M,...

For Retail Rocket, please refer to first two sections from the [README for the dataset](https://github.com/sfhorng/AIPI-531-Final-Project/tree/main/Retail_Rocket_Data).

## Evaluation Metrics and Results
The offline evaluation metrics used are Hit Ratio (HR@k) and Normalized Discounted Cumulative Gain (NDCG@k). HR@k is the ratio of ground-truth items that are in the top k slots of the recommendation ranking. NDCG@k represents how relevant the top k ranked items are, and a higher score indicates that relevant products are given higher rankings.

The following results are for purchases.
### Results for the H&M Dataset
#### Hit Ratio at K
|  HR@5 | HR@10 | HR@15 | HR@20 |
| --- | --- | --- | --- |
|     |     |     |     |

####  Normalized Discounted Cumulative Gain at K
|  NG@5 | NG@10 | NG@15 | NG@20 |
| --- | --- | --- | --- |
|     |     |     |     |

### Results for the Retail Rocket Dataset
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