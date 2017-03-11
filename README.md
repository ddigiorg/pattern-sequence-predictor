# Pattern & Sequence Predictor

Pattern & Sequence Predictor (PSP) is a machine intelligence architecture based on intelligence principles of the mammalian neocortex and is inspired by research from Numenta's Hierarchical Temporal Memory (HTM) and Ogma's Feynman Machine (FM).  PSP learns sparse distributed neuron activations observed from spatial-temporal sensory-motor  and encoded from an input space.

## Theory

"The beginning of wisdom is the definition of terms" is a quote attributed to Athenian philosopher Socrates.  In order to understand the principles of Machine Intelligence, one must first define intelligence and its related concepts: the environment where an intelligence exists and the knowledge intelligence uses.

### Intelligence

Intelligence exists in an ordered environment and acquires knowledge in an attempt to achieve a wide range of goals. 

Something that is intelligent [Ref]:

continuously learns
interacts with the environment
is general in nature
has goal-oriented behavior

### Environment

Intelligent beings exist in a spatial-temporal environment called the Universe, all of time and space and its contents.  The universe is governed by natural laws the sciences seek to understand (gravity, nuclear forces, electromagnetism, etc.) which provide order: repeatable events or concepts providing predictability.  For example, Earth revolves around the Sun in an ellipse taking 365 days for a single orbit.  The fact that Terra does not with chaotic intent decide to careen away from the warm loving embrace of Sol into the cold dark void is a blessing of order.  Extending the concept of chaos, the Earth does not explode into an unpredictable soup of inter- and extra-universal possibilities but rather remains relatively stable.  We can rest comfortably predicting Earth will revolve around the Sun as long as the natural order, understood or not, remains stable. 

### Knowledge

Fundamentally, knowledge is observed sequences of events or concepts.  In the spatial-temporal environment intelligent creatures observe through sensors (eyes, ears, noses, tounges, nerves, etc.) and influence through motors (muscles).  


By observing the repeated spatial-temporal events of environment over time


Important events that happen that the intelligence failed to predict recieve attention and learning.

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

### Sensory-Motor

Intelligent creatures have inputs and outputs.

### Sparse Distributed Representation (SDR) 

not all neurons are on at once.  In fact the amount of active neurons at a given moment is very sparse.

### On-line learning

no backprop, learn-as-you-go, faster

## Functions

PSP has 4 core functions: Spatial Encoding, Temporal Encoding, Decoding, and Learning.

### Spatial Encoding

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

### Temporal Encoding (Prediction)

### Decoding

### Learning

## Improvements
- Improve pattern and sequence recognition by chunking setPatternSums and setSequenceSums
- Change OpenCL images to buffers for much larger space allocation
- Fix learnInputMemories by having each neuron learn to recognize different field values
- Once learnInputMemories fixed, make 2d ball physics demo



## To-Do list
- Finish README
- Implement a "forecast" function to show running future predictions
- Show a very simple moving square demo
