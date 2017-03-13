# Divination Machine

Divination Machine (DM) is a Machine Intelligence architecture based on intelligence principles of the mammalian neocortex and is inspired by research from Numenta's Hierarchical Temporal Memory (HTM) and Ogma's Feynman Machine (FM).  Although its name implies the invocation of the supernatural, DM's functionality is obviously mundane.  Divination Machine:

- Observes a spatial-temporal sensory-motor input space
- At each moment in time, encodes inputs into sparsly distributed neuron activations
- Learns spatial and temporal encodings and stores them in memory
- Predicts a possible future output space based on memories

## Philosophy

"The beginning of wisdom is the definition of terms" -Socrates.  

To develop Machine Intelligence one must first define "intelligence" and its principles.  Unfortunately philosophers and scientists do not agree on a single definition of intelligence, but most would agree humans are a species with intelligence capabilities.  Therefore, the mammalian neocortex and its experience with the world gives us some guidelines on how to explore intelligence.   

### Intelligence

Intelligence exists in an ordered environment and acquires knowledge in an attempt to achieve goals. 

### Ordered Environment

The ultimate model of reality for an intelligence is the environment it percieves.  We call it "Universe", all of time and space and its contents.  Thus far our Universe is governed by natural laws, human conceptions of gravity, nuclear forces, electromagnetism, etc., which provide order, reoccouring concepts which are therefore predictable.   For example, Earth revolves around the Sun in an ellipse taking 365 days for a single orbit.  Earth does not randomly careen away from the Sun or explode into a soup of ever-changing infinite impossibilities (like unpredictability of static on a TV).  Therefore we can predict Earth will exist as-is and revolve around the Sun as long as the natural order, understood or not, remains stable.  Thus to seek an understanding of intelligence, we must have a concept of an intelligence's spatial-temporal perception of the Universe.

Space:  A spatial phenomenon or "pattern" is a n-dimensional extent in which objects occur and have relative position.  Examples include:
- Visual: the human retina is a 2d plane of photoreceptors that respond to red, green, and blue light frequencies.
- Auditory: the human cochlea is a 1d line (structurally a 2d spiral) with hairs that respond to vibrations induced by sound frequencies.
- Motor: 

Time:  A temporal phenomenon or "sequence" is a 1-dimensional 1-directional progression of spatial patterns.  Examples include:
- Visual: Watching a ball fall from the sky.
- Autitory
- Somatosensory
- Motor

### Knowledge

Intelligence aquires knowledge, sequences of patterns, by observing its environment through sensors (eyes, ears, noses, tounges, nerves, etc.) and influencing its environment through motors (muscles, vocal cords, etc).  Knowledge is stored in memory, learning, and intelligence uses what it has learned to make predictions about reoccouring events and influence events so that future events may occour.

### Perspective

All known intelligences thus far have limited perspective, how much an intelligence observes and how much it can remember.  Humans can not see and store all matter at all angles at all time and even if we could, the visible spectrum is just a small range of possible frequencies.  Thus, the ultimate promise of Machine Intelligence is being able to develop intelligences with greater or different perspectives.

## DM Architecture

The mammalian neocortex facilitates the concepts of intelligence discussed above.

### Cortical Columns

### Sparsly Distributed Neurons

not all neurons are on at once.  In fact the amount of active neurons at a given moment is very sparse.

### On-line learning

no backprop, learn-as-you-go, faster

## DM Functions

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

## Code Improvements
- Improve pattern and sequence recognition by chunking setPatternSums and setSequenceSums
- Change OpenCL images to buffers for much larger space allocation
- Fix learnInputMemories by having each neuron learn to recognize different field values
- Once learnInputMemories fixed, make 2d ball physics demo



## To-Do list
- Finish README
- Implement a "forecast" function to show running future predictions
- Show a very simple moving square demo
