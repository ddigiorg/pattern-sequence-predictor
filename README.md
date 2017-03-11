# Pattern & Sequence Predictor

Pattern & Sequence Predictor (PSP) is a machine intelligence architecture based on intelligence principles of the mammalian neocortex and is inspired by research from Numenta's Hierarchical Temporal Memory (HTM) and Ogma's Feynman Machine (FM).  PSP learns sparsly distributed neuron activations observed and encoded from spatial-temporal sensory-motor input space.  Additionally, PSP predicts a possible future output space based on the architecture's memories.

## Theory

"The beginning of wisdom is the definition of terms" is a quote attributed to Athenian philosopher Socrates.  In order to understand the principles of Machine Intelligence, one must first define intelligence and its related concepts: the environment where an intelligence exists and the knowledge intelligence uses.

Ultimately one of the greatest promises of Machine Intelligence is not in domains familiar to humans like vision and hearing, but in abstract domains like radio waves, internet data, and .

### Intelligence

Intelligence acquires knowledge in an attempt to achieve goals. 

Human-like intelligence:
- observes and interacts with an ordered environment
- continuously learns (acquires knowledge)
- is general in nature (attempts to achieve a wide range of goals)
- has goal-oriented behavior

### Environment

Environment is governed by the spatial-temporal natural order of concepts or events.

Intelligent beings exist in a environment called the Universe, all of time and space and its contents.  The universe is governed by natural laws the sciences seek to understand (gravity, nuclear forces, electromagnetism, etc.) which provide order: repeatable concepts or events which are predictable.  For example, Earth revolves around the Sun in an ellipse taking 365 days for a single orbit.  The fact that Terra does not, with chaotic intent, decide to careen away from the warm loving embrace of Sol into the cold dark void is a blessing of order.  Extending the concept of chaos, Earth does not explode into a soup of matter-energy and infinite impossibilities like the random unpredictability of static on a TV.  We can rest comfortably assured Earth-matter will remain and revolve around the Sun as long as the natural order, understood or not, remains stable.

A spatial phenomenon or "space" is a n-dimensional extent in which objects occur and have relative position.  Examples include:
- Visual: the human retina is a 2d plane of photoreceptors that respond to red, green, and blue light frequencies.
- Auditory: the human cochlea is a 1d line (structurally a 2d spiral) with hairs that respond to vibrations induced by sound frequencies.
- Somatosensory: 

A temporal phenomenon or "time" is a 1-dimensional 1-directional sequence of spatial events.  Examples include:
- Visual
- Autitory
- Somatosensory
- Motor

### Knowledge

Knowledge is sensory and/or motor sequences of concepts or events.

In a spatial-temporal environment, an intelligence observes through sensors (eyes, ears, noses, tounges, nerves, etc.) and influences through motors (muscles, vocal cords, etc).  By experiencing and learning knowledge, the intelligence, if faced with a similar experience, may use its memories to predict a future outcome.  Then the intelligence may use its motor functions to influence the environment to achieve a desired outcome.  For example, say a friend throws a ball to you.  


Important events that happen that the intelligence failed to predict recieve attention and learning.

## Architecture

The mammalian neocortex facilitates the concepts of intelligence discussed above.

### Cortical Columns

### Sparsly Distributed Neurons

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
