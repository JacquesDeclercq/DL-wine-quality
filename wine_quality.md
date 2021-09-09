# Deep learning - Wine Quality challenge

- Repository: `DL-wine-quality`
- Type of Challenge: `Learning`
- Duration: `3 days`
- Deadline: `dd/mm/yy H:i AM/PM`
- Deployment strategy :
  - GitHub page
- Team challenge : `solo`

![NYC Picture (Image)](https://www.wine-searcher.com/images/news/74/12/faves1-10007412.jpg)

## Mission objectives

- Use a deep learning library
- Prepare a data set for a machine learning model
- Put together a simple neural network
- Tune parameters of a neural network

## The Mission

Wine tasting has been around since the creation of wine itself. However, in the modern era, more importance has been given to drinking a _good_ wine, e.g. a French _Bordeaux_.
France has always been hailed as the _land of the wine_.
However, during the [Judgment of Paris](<https://en.wikipedia.org/wiki/Judgment_of_Paris_(wine)>) in 1976, a Californian wine scored better than a French wine which led to the increase in popularity of Californian wine.

Moreover, it has been shown that there are [many biases in wine tasting](https://en.wikipedia.org/wiki/Blind_wine_tasting).

That is why we put together this project to let an AI predict the quality of a wine.

### Must-have features

- Use `pytorch` or `keras` to build the model.
- The model is able to train.
- A baseline model is built using a simple neural network and the original dataset.
- The code is formatted using `black`.

### Miscellaneous information

The dataset `wine.csv` has already been downloaded for you in [../../additional_resources/datasets/Wine Quality/wine.csv](../../additional_resources/datasets/Wine%20Quality/wine.csv).

The objective of this project is mainly to get your hands dirty with `pytorch` or `keras`. **Choose the one that you prefer to do the project.**
The objective is not to get the best possible model.

You can reuse the environments created with the package information in `pytorch_requirements.txt` or `tf_requirements.txt`. It will make it much easier.

The dataset is already cleaned and has no missing values. You can do whatever you want to increase the score of the model.
There are multiple things to try:

- Define the problem as a classification problem
- Define the problem as a regression problem
- Feature engineering
- Feature normalization
- Resampling
- Hyper-parameter tuning
  - Change the learning rate
  - Change the loss function
  - etc...
- Modify the architecture
  - Number of layers
  - Number of neurons per layer
  - Activation functions
  - etc...
- etc...

In machine learning, it is customary to have a [baseline model](https://blog.insightdatascience.com/always-start-with-a-stupid-model-no-exceptions-3a22314b9aaa).
**Thus, the first step of this project is to get a very simple architecture to work with the standard, unmodified data.**

**Then** you can do all the above to get better performance.

## Deliverables

1. Publish your source code on the GitHub repository.
2. Pimp up the README file:
   - Description
   - Installation
   - Usage
   - (Visuals)
   - (Contributors)
   - (Timeline)
   - (Personal situation)

### Steps

1. Create the repository.
2. Study the request (What & Why ?)
3. Identify technical challenges (How ?)

## Evaluation criteria

| Criteria       | Indicator                                                                 | Yes/No |
| -------------- | ------------------------------------------------------------------------- | ------ |
| 1. Is complete | There is a published GitHub page available.                               |        |
| 2. Is Correct  | There are no warnings or errors when running the script.                  |        |
|                | The code is well documented.                                              |        |
|                | The code is formatted using `black`.                                      |        |
|                | The student has used either `keras` or `pytorch` to build the model.      |        |
|                | The model compiles.                                                       |        |
|                | The loss and accuracy of the model changes over the training procedure.   |        |
|                | There is a baseline model to compare future improvements with.            |        |
| 3. Is great    | The student has tried many techniques to increase the score of the model. |        |

## A final note of encouragement

You've been waiting for this, and I'm certain that you are ready for it.

![You've got this!](https://media.giphy.com/media/ctNDDU3a4ffK1su6yJ/giphy.gif)
