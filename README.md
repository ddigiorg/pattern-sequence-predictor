# Pattern & Sequence Predictor

Pattern & Sequence Predictor (PSP) is a machine intelligence architecture based on intelligence principles of the mammalian neocortex and inspired by research from Numenta's Hierarchical Temporal Memory (HTM) and Ogma's Feynman Machine (FM).  PSP learns and predicts spatial-temporal sensory-motor patterns from sparse distributed neuron activations observed and encoded from an input space.

## Theory

The idea is over time shit repeats itself.  We can see this shit and build up experience knowing that this shit happend.  Therefore if we see similar shit happening, we can predict that similar shit will happen. 


If someone throws you a ball and you wish to catch it, your brain does not calculate the ball's trajectory via Newtonian physics and does not calculate the required arm position to make the catch via inverse kinematics.  Rather your brain builds up a familiarity or intuition, or "muscle memory", about the ball's movements and your arm's movements by observing them move through time.

This intuition comes from Taken's Theorum, a principle of Dynamical Systems and Chaos Theory.  Taken's theorum states that if you observe one parameter of a state-space over time you can build a predictive model, or manifold, of the state-space.  In our environment we have visible states like green grass, blue skies, a friend, and a ball traveling towards you through the air.  However our universe has invisible states (or order) we call gravity, physics, electromagnitism, etc.  Taken's theorum simply states by seeing our environment over time we build an understanding of hidden causal relationships, i.e. the common proverbial phrase "what goes up, must come down".  By using our past experiences of a ball moving through time and seeing the ball move we can safely assume it will fall down to the earth.

Of course this principle extends beyond falling balls.  Any form of data viewed over time can build an internal predictive intuition of the world.

Due to the seemingly random and complex nature of dynamical systems, only short term predictions are accurate.

## Principles

### Spatial-Temporal

Physicists and science fiction often talk about "the space-time continuum",  mathematical model that combines space and time into a single interwoven continuum.  Our brains operate in a "spatial-temporal" environment.

A spatial phenomenon or "space" is a n-dimensional extent in which objects occur and have relative position.  Human sensors and actuator spatial examples include:
- Visual: the human retina is a 2d sheet of photoreceptors that respond to red, green, and blue light intensity.
- Auditory: the human cochlea is a 1d line (structurally a 2d spiral) with hairs that respond to vibrations induced by sound frequency.
- Somatosensory: 
- Motor: 

A temporal phenomenon or "time" is a 1-dimensional 1-directional sequence of spatial events.  Human sensors and actuator temporal examples include:
- Visual
- Autitory
- Somatosensory
- Motor

#### Sensory-Motor

Intelligent creatures have inputs and outputs.

#### Sparse Distributed Representation (SDR) 

not all neurons are on at once.  In fact the amount of active neurons at a given moment is very sparse.

#### On-line learning

no backprop, learn-as-you-go, faster

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
