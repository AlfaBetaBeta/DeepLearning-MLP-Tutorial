# FashionMNIST Image Classification

This repository comprises the noteboook `MLP_Tutorial.ipynb`, embedding a tutorial template that belongs originally to TensorFlow, alongside additional contents with relevant elaborations. The template roughly corresponds to Section `I. Basic classification` in the notebook. It showcases the assembly and training of an image classifier via `tensorflow.keras` based on the [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset, which embeds 70,000 low resolution (28 x 28 pixel) images of clothing articles from 10 categories, as per the sample below (each category takes three rows).

<img src="https://github.com/AlfaBetaBeta/DeepLearning-MLP-Tutorial/blob/master/img/FashionMNIST.png" width=100% height=100%>

The classifier is a multi-layer perceptron (MLP) model, with one hidden layer comprising 144 neurons and `relu` activation function. Consistently with the category levels, the output layer consists of 10 neurons with `softmax` activation. The default `GlorotUniform` initialiser has been fed a `RANDOM_SEED` to ensure reproducibility of results. For reference, the summary of the MLP model is shown below.

<img src="https://github.com/AlfaBetaBeta/DeepLearning-MLP-Tutorial/blob/master/img/model-summary.png" width=55% height=55%>

The additional elaborations in the notebook are mostly located from Section `II. Post-processing` onwards, regarding the following aspects, succinctly presented here:

Toc

### Weight interpretation

It is of interest to retrieve the trained weights associated to the hidden layer (i.e. `model.layers[1]`) to assess if they can be visually interpreted. This can be achieved via the method `get_weights()` and some convenient reshaping. Illustratively, gathering all weights associated to the **first** neuron in the hidden layer and rearranging as a (28 x 28) array leads to the following representation:

<img src="https://github.com/AlfaBetaBeta/DeepLearning-MLP-Tutorial/blob/master/img/Layer1-Weight-Unit1.png" width=35% height=35%>

There is no salient interpretation, but it was expected that a number of neurons would learn a pattern that would not necessarily be intuitive. If the same process is repeated iteratively over all neurons in the hidden layer, the following array of arrays is retrieved:

<img src="https://github.com/AlfaBetaBeta/DeepLearning-MLP-Tutorial/blob/master/img/Layer1-Weight-AllUnits.png" width=100% height=100%>

Amongst the 144 neurons, there are some patterns that are intuitive (to a human observer) indeed, as shown below.

<img src="https://github.com/AlfaBetaBeta/DeepLearning-MLP-Tutorial/blob/master/img/Layer1-Weight-Subset.png" width=100% height=100%>

Note that neuron 129 seems to have learned patterns pointing to two categories simultaneously (possibly `Sneaker` and `Shirt`).

Formally, the weights associated to the output layer (`model.layers[2]`) can be treated in the same manner. When doing so, each set of weights transforms into a (12 x 12) array.

<img src="https://github.com/AlfaBetaBeta/DeepLearning-MLP-Tutorial/blob/master/img/Layer2-Weight-AllUnits.png" width=100% height=100%>

In this case, there is no intuitive interpretation for these weights.


### Handling of a blank input

It also bears interest to inspect what is the output of the trained MLP when passing a blank image as input (i.e. an array `np.ones((28,28))` as the one below, recalling the pixel values have been normalised).

<img src="https://github.com/AlfaBetaBeta/DeepLearning-MLP-Tutorial/blob/master/img/Blank-input.png" width=20% height=20%>

Under such circumstances, the model prediction is distinctly `Bag`:

<img src="https://github.com/AlfaBetaBeta/DeepLearning-MLP-Tutorial/blob/master/img/Blank-input-classification.png" width=20% height=20%>

As a double check, the forward propagation can be replicated manually by executing the following steps:

1. retrieve the weight (`model.layers[1].get_weights()[0]`) and bias (`model.layers[1].get_weights()[1]`) coefficients of the hidden layer and perform the appropriate array operations with the blank input.

2. apply the `relu` activation function on the previous result to obtain the hidden feature values.

3. calculate the 'raw' output values by operating on the previous hidden feature values and the weight and bias coefficients of the output layer.

4. apply the `softmax` activation function on the previous result.

These steps lead to exactly the same array of predictions as the one resulting from feeding a blank input to `model.predict()`, as expected. When rearranging the hidden feature values from step 2 above as a (12 x 12) array, the following is obtained:

<img src="https://github.com/AlfaBetaBeta/DeepLearning-MLP-Tutorial/blob/master/img/Blank-input-hidden-layer-values.png" width=20% height=20%>

There is no intuitive visual meaning. By inspection (though this can be corroborated with ease), the neuron with the largest value after activation is neuron number 33 which, upon review of the corresponding weight distribution in the weight interpretation elaboration, does not convey intuitive meaning either. 

Also, and leaving the bias coefficients aside, only 30 of the 144 neurons in the hidden layer contribute to the `Bag` neuron when feeding in a blank input. The fact that `Bag` is predicted from a blank is seemengly a serendipitous byproduct of the parameter values learned at training.


### Weight evolution over training

By defining a pair of convenient functions to retrieve the weight distribution after each epoch during training, it is possible the track the evolution of the weight coefficients, and possibly enable visual inspection of the weight array to check if any patterns emerge during training (in this case defined by `n_epochs=15`). Illustratively, doing this for the **first** neuron of the hidden layer returns:

<img src="https://github.com/AlfaBetaBeta/DeepLearning-MLP-Tutorial/blob/master/img/Layer1-Weight-Unit1-evolution-training.png" width=20% height=20%>

Interestingly, the weight distribution of this neuron adopts an intuitive shape early on (resemblance to class `Coat` or `Shirt` at epoch 4) but then becomes rather 'abstract' again. It is suggested that this could be interpreted as another sign of overfitting, though more thorough research would be needed to confirm this.






