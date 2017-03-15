# Divination Machine

Divination Machine (DM) is a Machine Intelligence architecture based on intelligence principles of the mammalian neocortex and is inspired by research from Numenta's Hierarchical Temporal Memory (HTM) and Ogma's Feynman Machine (FM).  Although its name implies the invocation of the supernatural, DM's functionality is obviously mundane.  Divination Machine:

- Observes spatial-temporal sensory-motor input
- At each moment in time, encodes inputs into sparsly distributed neuron activations
- Learns spatial and temporal encodings and stores them in memory
- Predicts a possible future output based on memories

## Philosophy

"The beginning of wisdom is the definition of terms" -Socrates.  

To develop Machine Intelligence one must first define "intelligence" and its principles.  Unfortunately philosophers and scientists do not agree on a single definition of intelligence, but most would agree the human species has intelligence.  Therefore, the mammalian neocortex and how humans and animals experience the world gives us some guidelines on how to explore intelligence.   

### Intelligence

Intelligence exists in an ordered environment and acquires knowledge in an attempt to achieve goals.

### Ordered Environment

The ultimate model of reality for an intelligence is the environment it percieves.  We call it "Universe", all of time and space and its contents.  Thus far our Universe is governed by natural laws, human conceptions of gravity, nuclear forces, electromagnetism, etc., which provide order, reoccouring concepts which are therefore predictable.   For example, Earth revolves around the Sun in an ellipse taking 365 days for a single orbit.  Earth does not randomly careen away from the Sun or explode into a soup of ever-changing infinite impossibilities (like unpredictability of static on a TV).  Therefore we can predict Earth will exist as-is and revolve around the Sun as long as the natural order, understood or not, remains stable.  Thus to seek an understanding of intelligence, we must have a concept of an intelligence's spatial-temporal perception of the Universe.

#### Space

A spatial phenomenon or "pattern" is a n-dimensional extent in which objects occur and have relative position.  Examples include:
- Visual: the human retina is a 2d plane of photoreceptors that respond to red, green, and blue light frequencies.
- Auditory: the human cochlea is a 1d line (structurally a 2d spiral) with hairs that respond to vibrations induced by sound frequencies.
- Motor: 

#### Time

A temporal phenomenon or "sequence" is a 1-dimensional 1-directional progression of spatial patterns.  Examples include:
- Visual: Watching a ball fall from the sky.
- Autitory
- Somatosensory
- Motor

### Knowledge

Knowledge, or experience, is simply sequences of patterns.  Intelligence aquires knowledge by observing its environment through sensors (eyes, ears, noses, tounges, nerves, etc.) and influencing its environment through motors (muscles, vocal cords, etc).  Knowledge is stored in memory, through a process called "learning".  If the intelligence observes a portion of a sequence, it can predict the upcomming patterns in the sequence will occour.  Therefore, intelligence uses what it has learned to make predictions about the future and can influence the environment so that desired events may occour.

However, the neocortex does not operate with colors, sounds, or tastes.  Fundamentally, a pattern in the neocortex is a set of neurons that activate in responce to an input at a certain time, or a spatial input.  For example, let's say you see a blue circle.  A set of neurons in your neocortex will activate in response to the visual stimulus of seeing the blue circle.  If the circle were suddenly to become red, another set of neurons would respond.  A sequence in the neocortex is therefore a sequential progression of neuron activations.  By converting, or encoding, observed patterns into neuron activations, the neocortex has a unified standard for storing and operating on knowledge.

For example:

[INSERT PICTURE HERE]

### Perspective

All known intelligences thus far have limited perspective, how much an intelligence observes and how much it can remember.  Humans can not see and store all matter at all angles at all time and even if we could, the visible spectrum is just a small range of possible frequencies.  Thus, the ultimate promise of Machine Intelligence is being able to develop intelligences with greater or different perspectives.

## DM Functions

Divination Machine has 4 core functions:
1. Spatial Encoding
2. Temporal Encoding
3. Decoding
4. Learning.

### Spatial Encoding

A spatial encoding is a pattern of neurons activated from an observed input at a single moment in time.  At each time step Spatial Encoding:
1. Converts the input into a Sparse Distributed Representation (SDR) of neuron activations called "Column Winners"
2. Searches "Pattern Memories" for "Column Winners" and if it exists returns the memory index, or "Spatial Encoding"

#### Step 1

A column, commonly refered to as a cortical "minicolumn" in neuroscience, is a group of neurons that share a receptive field, or a specific region, of the input space.  In DM each column selects a winner neuron by looking at every neuron in a column comparing it's memories to the column's receptive field input values.  The best matching neuron is the winner neuron of the column.  Programatically, each column essentially behaves like a Self Organizing Map where each neuron has a "sum" value calculated by taking the Euclidian distance of every node's memory and corresponding input value.

The Euclidian distance compares two sets of values and computes how similar they are to each other.  A shorter distance means the values are more similar while a larger distance means the values are less similar.  For each column, the node with the smallest distance is the winner node, the node who's memories are the most similar to the column's receptive field input values.

![alt tag](https://raw.githubusercontent.com/ddigiorg/neuroowl.github.io/master/images/technology/divination_machine/spatial_encoding_figure_1.png)

In the above figure, the receptive field of a column is a 3x3 grid of monochromatic pixels.  For reference, a monochromatic color uses a single channel (i.e. red) of the 4 channel color representation (red, green, blue, and alpha).  Each pixel is a floating point value between 0.0f and 1.0f representing color intensity.  If we define our color sensitivity to be 0.001f there are 1,001 different color values, or intensities, represented in each pixel.  A 3x3 grid has 9 pixels in total which has 1,001^9, or ~1.01x10^27 unique possible color inputs at a single moment in time.  This is more than twice the estimated amount of neurons in the human brain (about 100 billion neurons)!

Spatial Sensitivity:  Intuitively if a human sees a 9x9 field of red with just 1 pixel just slightly less red we'd still recognize it as "a sea of red".  If that one pixel were to keep losing intensity, eventually the human brain would be able to recognize it as "a sea of red with a less red dot". 

For example, a column of 10 neurons will remember the top 10 most observed input pixel values.

#### Step 2

DM then searches its "Pattern Memory" to see if "Column Winners" SDR exists.  If it doesn't exist DM adds the SDR to its Pattern Memory.  If it exists then the index where it exists is called the "Spatial Encoding", a single integer value representing the spatial context observed input.

![alt tag](https://raw.githubusercontent.com/ddigiorg/neuroowl.github.io/master/images/technology/divination_machine/spatial_encoding_figure_2.png)

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
