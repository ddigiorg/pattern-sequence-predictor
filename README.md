# Divination Machine

Divination Machine (DM) is a Machine Intelligence architecture based on intelligence principles of the mammalian neocortex and is inspired by research from Numenta's Hierarchical Temporal Memory (HTM) and Ogma's Feynman Machine (FM).  Although its name implies the invocation of the supernatural, DM's functionality is obviously mundane.  Divination Machine:

- Observes spatial-temporal sensory-motor input
- At each moment in time, encodes inputs into sparsly distributed neuron activations
- Learns spatial and temporal encodings and stores them in memory
- Predicts a possible future output based on memories

## Philosophy

"The beginning of wisdom is the definition of terms" -Socrates.  

To develop Machine Intelligence one must first define "intelligence" and its principles.  Unfortunately philosophers and scientists do not agree on a single definition, but most would admit the human species is intelligent.  Therefore, the mammalian neocortex and how it experiences the world gives us some guidelines.  This project's definition of intelligence is:

Intelligence, existing in an ordered environment, acquires knowledge in an attempt to achieve goals.

### Ordered Environment

The ultimate model of reality for an intelligence is the environment it percieves.  We call it "Universe", all of time and space and its contents.  Thus far our Universe is governed by natural laws, human conceptions of gravity, nuclear forces, electromagnetism, etc., which provide order, reoccouring concepts which are therefore predictable.  For example, Earth revolves around the Sun in an ellipse taking 365 days for a single orbit.  Earth does not randomly careen away from the Sun or explode into a soup of ever-changing infinite impossibilities (like the unpredictability of static on a TV).  Therefore we can predict Earth will exist as-is and revolve around the Sun as long as the natural order, understood or not, remains stable.  

### Perspective

Although many of these principles of order can not be directly observed, they are known to exist through their effects.  In essence by observing the environment an intelligence gains an intuition for the underlying order of the environment.  Because we live in a space-time Universe, to seek an understanding of intelligence we must have a concept of an intelligence's spatial-temporal perception of its environment.

#### Space

A spatial phenomenon or "pattern" is a n-dimensional extent in which objects occur and have relative position.  Examples include:
- Visual: the human retina is a 2d plane of photoreceptors that respond to red, green, and blue light frequencies.
- Auditory: the human cochlea is a 1d line (structurally a 2d spiral) with hairs that respond to vibrations induced by sound frequencies
- Proprioception: the angle of a joint relative to two pieces of the limb

#### Time

A temporal phenomenon or "sequence" is a 1-dimensional 1-directional progression of spatial patterns.  Examples include:
- Visual: Seeing a ball fall from the sky
- Autitory: Hearing the changing pitch in a song
- Proprioception: Feeling the speed at which a joint moves

All known intelligences thus far have limited perspective: how much an intelligence observes and how much it can remember.  Humans can not see and store all matter at all angles at all time and even if we could, the visible spectrum is just a small range of possible frequencies.  Thus, the ultimate promise of Machine Intelligence is being able to develop intelligences with greater or different perspectives.

### Knowledge

Knowledge, or experience, is simply sequences of patterns.  Intelligence aquires knowledge by observing its environment through sensors (eyes, ears, noses, tounges, nerves, etc.) and influencing its environment through motors (muscles, vocal cords, etc).  Knowledge is stored in memory, through a process called "learning".  If the intelligence observes a portion of a sequence, it can predict the upcomming patterns in the sequence will occour.  Therefore, intelligence uses what it has learned to make predictions about the future and can influence the environment so that desired events may occour.

![alt tag](https://raw.githubusercontent.com/ddigiorg/neuroowl.github.io/master/images/technology/divination_machine/knowledge_figure_1.png)

However, the neocortex does not operate with colors, sounds, or tastes.  Fundamentally, a pattern in the neocortex is a set of neurons that activate in responce to an input at a certain time, or a spatial input.  For example, let's say you see a blue circle.  A set of neurons in your neocortex will activate in response to the visual stimulus of seeing the blue circle.  If the circle were suddenly to become red, another set of neurons would respond.  A sequence in the neocortex is therefore a sequential progression of neuron activations.  By converting, or encoding, observed patterns into neuron activations, the neocortex has a unified standard for storing and operating on knowledge.

![alt tag](https://raw.githubusercontent.com/ddigiorg/neuroowl.github.io/master/images/technology/divination_machine/knowledge_figure_2.png)

In the above graphic, pink and cyan circles as well as "wait" and "eat!" actions have been encoded into neuron activations.

## DM Functions

Divination Machine has 4 core functions:
1. Spatial Encoding
2. Temporal Encoding
3. Decoding
4. Learning

### Spatial Encoding

Spatial Encoding observes input at a single time step and attempts to recognize a learned pattern.  The algorithm uses neuron activations to group together similar patterns from the near infinite possibilites of an input space.  For example, say you are looking at a 3x3 area of red pixels, but 1 pixel slightly less red.  The human brain is unable to distinguish the slightly off-color pixel from the others.  Now if the off-color pixel were to keep losing intensity, eventually the human brain would be able to see a dark-red pixel standing apart from the other red pixels.  The human brain therefore has a certain level of pattern recognition sensitivity because it is limited by the number of neurons it has to recognize patterns.  

At each time step Spatial Encoding:
1. Converts the input into a Sparse Distributed Representation (SDR) of neuron activations called "Column Winners"
2. Searches "Pattern Memories" for "Column Winners" and if it exists returns the memory index, or "Pattern"

#### Columns of Neurons

A column, commonly refered to as a cortical "minicolumn" in neuroscience, is a group of neurons that share a receptive field, or a specific region, of the input space.  A single neuron in a column will activate in response to a range of similar receptive field input values.  Therefore, the more neurons in a column the more spatial sensitivity a column has to observe distinct patterns in an input.  

Say you are observing a 3x3 grid of monochromatic pixels on a computer screen.  For reference, a monochromatic color uses a single channel (i.e. red) of the 4 channel color representation (red, green, blue, and alpha).  Let's define each pixel in computer memory as a floating point value between 0.0f and 1.0f representing color intensity.  If we define our color sensitivity to be 0.001f there are 1,001 different color values, or intensities, represented in each pixel.  A 3x3 grid has 9 pixels in total which has 1,001^9, or ~1.01x10^27 unique possible color inputs at a single moment in time.  This is more than twice the estimated amount of neurons in the human brain (about 100 billion neurons)!  Of course the brain can not afford that many neurons to represent just a 3x3 pixel space when the human retina has millions of photoreceptors.  This is why spatial encoding is important!

#### Step 1

In DM each column selects a winner neuron by looking at every neuron in a column comparing it's memories to the column's receptive field input values.  The best matching neuron is the winner neuron of the column.  Programatically, each column essentially behaves like a Self Organizing Map where each neuron has a "sum" value calculated by taking the Euclidian distance of every node's memory and corresponding input value.

The Euclidian distance compares two sets of values and computes how similar they are to each other.  A shorter distance means the values are more similar while a larger distance means the values are less similar.  For each column, the node with the smallest distance is the winner node, the node who's memories are the most similar to the column's receptive field input values.

![alt tag](https://raw.githubusercontent.com/ddigiorg/neuroowl.github.io/master/images/technology/divination_machine/spatial_encoding_figure_1.png)

#### Step 2

Once DM has a set of neuron activations, "Column Winners", the algorithm searches its "Pattern Memory" to see if "Column Winners" exists.  If it doesn't exist DM adds the SDR to its Pattern Memory(see learning).  If "Column Winners" exists in "Pattern Memories" then the index where it exists is called the "Pattern", a single integer value representing the observed input.

![alt tag](https://raw.githubusercontent.com/ddigiorg/neuroowl.github.io/master/images/technology/divination_machine/spatial_encoding_figure_2.png)

### Temporal Encoding (Prediction)

 Temporal Encoding observes patterns from Spatial Encoding through time and attempts to recognize a learned sequence.

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
