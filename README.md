## Practice for Bayesian Inference

This repository contains a simple physics simulation and a basic implementation of variational Bayesian inference. The simulation can process 100s of objects in parallel and is written in Pytorch. 

The model consists of 4 Gaussian distributions representing the initial conditions of the simulation, two for each dimension of position and two for each dimension of velocity. The parameters are learned to cause the objects to orbit the center of the screen by maximizing the ELBO using gradient descent.

ELBO is calculated using an approximation for likelihood and the entropy of the model. The likelihood of an orbit is approximated by summing, at each time step, the dot product of the normal of each object's velocity and the difference between its position and the center, squared.

Example of training with high entropy priors containing no helpful information. 

## ElBO over time
![](https://github.com/ttenneb/bayesian_practice/blob/master/ELBO%20RESULTS.png)

## Initial distribution before training
![](https://github.com/ttenneb/bayesian_practice/blob/master/Initial%20Trajectories.gif)

## Initial distrubution after training
![](https://github.com/ttenneb/bayesian_practice/blob/master/Learned%20Initial%20Coniditons.gif)
