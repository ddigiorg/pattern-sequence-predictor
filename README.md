# Sequence Memory & Predictor

The Sequence Memory Predictor (SMP) is a machine intelligence architecture based on theoretical intelligence principles of the mammalian neocortex and is inspired by research from Numenta's Hierarchical Tempral Memory (HTM) and Ogma's Feyneman Machine (FM).  SMP views, learns, and predicts spatio-temporal sensory-motor patterns.

## Objective

The objective of this project is 

## Theory



A friend tosses you a ball and you catch it.  The ball sails through the air with a trajectory mathematically modeled by Newtonian Physics (or General Relativity for the masochist).  To recieve the ball, your arm moves to the necessary position with each joint modeled through inverse kinematics.  To solve these mathematical problems, you must know the mass of the ball, its initial velocity, the force of gravity on Earth, the length of each portion of the arm, etc.  However we neither know the exact values of these parameters, 


However, the brain has an intuitive sense about the physics of the ball without a n.  A child with no familiarity of basic physics theorums and mathematical models still knows that a ball thrown up will fall back down.


The game of catch you and your friend played is modeled differently in the human brain through "muscle memory".  Since birth you have been constantly observing the environment through the sensors in your eyes, ears, and body.  

## Functionality

Columns with nodes

Sparse Distributed Representation (SDR) 

SMP has 4 core functions: Encode, Predict, Decode, and Learn.

Input data types are floating point values between 0.0f and 1.0f.  For the remainder of the readme we will use a 9x9 grid of input values representing monochromatic color strength.  For reference, a monochromatic color uses a single channel (i.e. red) of the 4 channel color representation (red, green, blue, and alpha).

##### Encode

At each time step SMP encodes the input space into a spatial pattern, a single value representing the observed input.  

Each column observes a specific receptive field, a region of the input space.  Each node in a column has memories corresponding to each receptive field input value.  Each node also has a "sum" value calculated by taking the Euclidian distance of every node's memory and corresponding input value:
```
   for every value-memory pair i:
      sum += (value[i] - memory[i])^2
      
   Note: the square root of the Euclidian distance is removed for computational simplicity
```

The Euclidian distance compares two sets of values and computes how similar they are to each other.  A shorter distance means the values are more similar while a larger distance means the values are less similar.  For each column, the node with the smallest distance is the winner.

which stores how well the node's memories correspond to the column's receptive field input values.  The node's sum is 

##### Predict

##### Decode

##### Learn


- Each column looks at a receptive field in the input data space and encodes repeatedly seen patterns into a single neuron activation, or "winner neuron".
- The addresses of winner neurons for every column are stored in pattern memories.  The index of the pattern is used for sequence memory.
- specific sequences of patterns are stored in sequence memory and predict memory.

#### To-Do list
- Finish README
- Implement a "forecast" function to show running future predictions
- Show a very simple moving square demo

#### Improvements
- Improve pattern and sequence recognition by chunking setPatternSums and setSequenceSums
- Change OpenCL images to buffers for much larger space allocation
- Fix learnInputMemories by having each neuron learn to recognize different field values
- Once learnInputMemories fixed, make 2d ball physics demo
