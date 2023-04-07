# Compose-Recompose-Images
A simple mlp project written in python which compose image and then recompose it again (No prepared functions or libraries are allowed for implementing the multilayer model)


# About This Project
Implementing a multilayer perceptron neural network with 3 layers :<br /><br />
I) input layer(original image) with n neurons<br />
II) one hidden layer(composed image) with m (m<n) neurons<br />
III) output layer(recomposed image) with n neurons<br />
This network uses a Train Set with 91 images in order to train perceptron and then test it by using Test Set with 5 images and changing some parameters. This evaluation is based on *Error* and *PSNR(Read Attached Paper)* <br />














Note : The small number of neurons in the hidden layer causes underfit due to the over-simplicity of the model and the inability to find an optimal algorithm, it causes high error rate in calculating the output , and requires a high number of iterations in order to learn the model.
On the other hand, a high number also causes complexity and calculation error because it quickly converges towards the solution with a low number of iterations and insufficient training of the model, it reaches a learning that is not optimal. The optimal answers are in the range between these two.

2.parameter: different initial values for weights and biases = try random values 3 times
(constants : batch size = 1, learning rate = 0.1, number of hidden layer neurons = 16)
I)Error = 14.3% , Number of Iterations = 35
II)Error = 19% , Number of Iterations = 24
III)Error = 19% , Number of Iterations = 25
Note : Since these weights and biases generated randomly, you may get different results. But we know weight matrix indicates the effectiveness of a particular input. The greater the weight of the input, the more it will affect the network and accelerate the activation function. On the other hand, the bias is like an added interval in a linear equation and is an additional parameter in the neural network that is used to adjust the output along with the weighted sum of the neuron's inputs and helps to control the value at which the activation function is triggered. At first, we place both of them randomly because we don't need any initial view of the importance of each neuron or interruption, so we may give too much weight to a less important neuron or vice versa, but gradually we improve the network. The effect of initial weight and bias cannot be ignored and affects the number of repetitions as well as the final error, but anyway, due to our little information at the beginning of the work; It is unavoidable.

3.parameter: learning rate = (Respectively) 0.01, 0.1, 0.2, 0.5, 0.9
For each learning rate our results are shown below :
I)learning rate = 0.01 : Error = 23.8% , Number of Iterations = 153
II)learning rate = 0.1 : Error = 9.5% , Number of Iterations = 24
III)learning rate = 0.2 : Error = 19% , Number of Iterations = 17
IV)learning rate = 0.5 : Error = 14.3% , Number of Iterations = 9
V)learning rate = 0.9 : Error = 28.6% , Number of Iterations = 13
Alt text

Note : Graph is actually discrete, but its points have plotted continuous.
Learning rate controls the speed at which the model adapts to the problem. Smaller learning rates require more number of iterations, due to smaller changes in the weights per update, while larger learning rates result in faster changes and require fewer iterations. High value of learning rate can cause quickly converge to a suboptimal solution, In the other hand low value of learning rate can cause the process to stuck.

Built With
python

Getting Started
Prerequisites
put Data in your project path
numpy
pip install numpy
matplotlib
pip install matplotlib



License
Distributed under the MIT License. See LICENSE.txt for more information


Contact
rezaie.somayeh79@gmail.com
