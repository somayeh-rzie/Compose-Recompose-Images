# Compose-Recompose-Images
A simple mlp project written in python which compose image and then recompose it again (No prepared functions or libraries are allowed for implementing the multilayer model)


# About This Project
Implementing a multilayer perceptron neural network with 3 layers :<br /><br />
I) input layer(original image) with n neurons<br />
II) one hidden layer(composed image) with m (m<n) neurons<br />
III) output layer(recomposed image) with n neurons<br /><br />

First of all we read our *.jpg* train files and store them in a 256 * 256 array.<br />
Then we partition our  256 * 256  pixels images to  8 * 8  blocks by using `reassample` function.<br />
Now we have two methods for training our model:<br />
- Standard Train : <br />
As we always train our models with backpropagation algorithm but with multiple networks each have its own weight and bias matrices and uniformly distribute blocks between networks.<br />
- Momentum Train : <br />
Here we have a parameter named momentum (first initialized 0.5).<br />
All steps are as same as Standard Train but:<br />
I) Our iteration condition is minimum psnr [^1]. that we set to 9.3 based on trial and error.<br />
$$Cost = \sum\sum (x_ij - \hat{x_ij})^2$$ <br />
$$\left( \sum_{k=1}^n a_k b_k \right)^2 \leq \left( \sum_{k=1}^n a_k^2 \right) \left( \sum_{k=1}^n b_k^2 \right)$$ <br />
II) For updating weights and biases we use values of 2 previous levels.<br />


Evaluation is based on *Error* and *PSNR* scales.<br /><br />

[^1]: Read Attached Paper


# Built With
- [python](https://www.python.org/) <br /><br />

# Getting Started
### Prerequisites
- put TestSet and TrainSet in your project path
- cv2 <br />
    `pip install opencv-python` <br />
- numpy <br />
    `pip install numpy` <br />
- statistics <br />
    `pip install statistics` <br />
- matplotlib <br />
    `pip install matplotlib` <br />

    
<br />

# License
Distributed under the MIT License. See `LICENSE.txt` for more information
<br /><br />

# Contact
rezaie.somayeh79@gmail.com
