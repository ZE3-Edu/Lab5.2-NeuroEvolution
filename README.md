# Lab5.2 - NeuroEvolution
Be sure to look through the Jupyter notebook before working on lab5_2.py!

## Overview
In this short lab, you'll start using evolution to find neural networks that solve problems for you. In the next homework assignment, you'll get to choose to use either neuroevolution or genetic programing to build controllers for a simple robot!

## What to do
In order to get this code working, you'll need to implement a mutation function to introduce variation into the neural networks. I've given you a tournament selection function, a relatively okay function to evaluate fitness, and the code to generate random neural networks with different structures. 

If you run this as is, you'll see the initial increase in fitness from evolution sorting through the variation in the random starting population. Once you implement the mutation function, you should see further improvements.

If you want to try a more challenging problem, think about how you might implement a [tick-tack-toe](https://en.wikipedia.org/wiki/Tic-tac-toe) *AI* that makes its move based on the state of the game board!