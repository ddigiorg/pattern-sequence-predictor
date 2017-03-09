# Pattern & Sequence Predictor

Pattern & Sequence Predictor (PSP) is a machine intelligence architecture based on intelligence principles of the mammalian neocortex and inspired by research from Numenta's Hierarchical Temporal Memory (HTM) and Ogma's Feynman Machine (FM).  PSP learns and predicts spatial-temporal sensory-motor patterns from sparse distributed neuron activations observed and encoded from an input space.

## Theory

dynamical systems

A friend tosses you a ball and you catch it.  The ball sails through the air with a trajectory mathematically modeled by Newtonian Physics (or General Relativity for the masochist).  To recieve the ball, your arm moves to the necessary position with each joint modeled through inverse kinematics.  To solve these mathematical problems, you must know the mass of the ball, its initial velocity, the force of gravity on Earth, the length of each portion of the arm, etc.  However we neither know the exact values of these parameters, 


However, the brain has an intuitive sense about the physics of the ball without a n.  A child with no familiarity of basic physics theorums and mathematical models still knows that a ball thrown up will fall back down.


The game of catch you and your friend played is modeled differently in the human brain through "muscle memory".  Since birth you have been constantly observing the environment through the sensors in your eyes, ears, and body.  

## Principles

spatial-temporal

Spatial: A "space" is a n-dimensional extent in which objects occur and have relative position.  Human examples include:
- Visual: the human retina is a 2d sheet of photoreceptors that respond to red, green, and blue light intensity.
- Auditory: the human cochlea is a 1d line (structurally a 2d spiral) with hairs that respond to vibrations induced by sound frequency.
- Somatosensory: 
- Motor: 

Temporal

sensory-motor


Columns with nodes

Sparse Distributed Representation (SDR) 

On-line learning

## Functions

PSP has 4 core functions: Spatial Encoding, Temporal Encoding, Decoding, and Learning.

#### Spatial Encoding

At each time step PSP encodes data from the input space into a "spatial pattern", a single value representing the observed input.



Take for example a 9x9 grid of monochromatic pixels, a visual spatial-sensory input space we will use to demonstrate encoding.  For reference, a monochromatic color uses a single channel (i.e. red) of the 4 channel color representation (red, green, blue, and alpha).  Each pixel is a floating point value between 0.0f and 1.0f representing color intensity.  If we define our color sensitivity to be 0.001f there are 1,001 different color values, or intensities, represented in each pixel.  A 9x9 grid has 81 pixels which has 1,001^81, or ~1.08x10^243 unique possible spatial-sensory inputs.  This is about twice the estimated atoms in the universe!

Spatial Sensitivity:  Intuitively if a human sees a 9x9 field of red with just 1 pixel just slightly less red we'd still recognize it as "a sea of red".  If that one pixel were to keep losing intensity, eventually the human brain would be able to recognize it as "a sea of red with a less red dot".  

The amount of nodes in a column corresponds to the spatial-sensitivity of the PSP.  A 


To achieve the architecture uses cortical columns of neurons that behave like Self Organizing Maps and observe a specific receptive field of the input space.  Each node in a column has memories corresponding to each receptive field input value.  Each node also has a "sum" value calculated by taking the Euclidian distance of every node's memory and corresponding input value:
```
   for every value-memory pair i:
      sum += (value[i] - memory[i])^2
      
   Note: the square root of the Euclidian distance is removed for computational simplicity
```

The Euclidian distance compares two sets of values and computes how similar they are to each other.  A shorter distance means the values are more similar while a larger distance means the values are less similar.  For each column, the node with the smallest distance is the winner node, the node who's memories are the most similar to the column's receptive field input values.

#### Temporal Encoding (Prediction)

#### Decoding

#### Learning

## Improvements
- Improve pattern and sequence recognition by chunking setPatternSums and setSequenceSums
- Change OpenCL images to buffers for much larger space allocation
- Fix learnInputMemories by having each neuron learn to recognize different field values
- Once learnInputMemories fixed, make 2d ball physics demo



## To-Do list
- Finish README
- Implement a "forecast" function to show running future predictions
- Show a very simple moving square demo
