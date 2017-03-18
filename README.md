<img src="https://raw.githubusercontent.com/ddigiorg/neuroowl.github.io/master/images/technology/divination_machine/logo.png" alt="DM Logo" width=100/>

# Divination Machine

Divination Machine (DM) is a Machine Intelligence architecture based on intelligence principles of the mammalian neocortex and is inspired by research from Numenta's Hierarchical Temporal Memory (HTM) and Ogma's Feynman Machine (FM).  Although its name implies the invocation of the supernatural, DM's functionality is obviously mundane.  Divination Machine:

- Observes spatial-temporal sensory-motor input
- At each moment in time, encodes inputs into sparsly distributed neuron activations
- Learns spatial and temporal encodings and stores them in memory
- Predicts a possible future output based on memories

For now DM can **observe** any time-series input sensory and/or motor data, **learn** patterns and sequences of patterns, and **predict** future patterns and sequences of patterns.  The ultimate goal of DM is to make motor **decisions** based on predictions learned from sensory input.  The DM architecture is still in its infancy and has lots of room for improvements and optimizations like the ones listed in this paper, but the underlying theory is simple and powerful.

## Philosophy and Theory

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

### The Cortex: Columns of Neurons

A column, commonly refered to as a cortical "minicolumn" in neuroscience, is a group of neurons that share a receptive field, or a specific region, of the input space.  A single neuron in a column will activate in response to a range of similar receptive field input values.  Therefore, the more neurons in a column the more spatial sensitivity a column has to observe distinct patterns in an input.  

Say you are observing a 3x3 grid of monochromatic pixels on a computer screen.  For reference, a monochromatic color uses a single channel (i.e. red) of the 4 channel color representation (red, green, blue, and alpha).  Let's define each pixel in computer memory as a floating point value between 0.0f and 1.0f representing color intensity.  If we define our color sensitivity to be 0.001f there are 1,001 different color values, or intensities, represented in each pixel.  A 3x3 grid has 9 pixels in total which has 1,001^9, or ~1.01x10^27 unique possible color inputs at a single moment in time.  This is more than twice the estimated amount of neurons in the human brain (about 100 billion neurons)!  Of course the brain can not afford that many neurons to represent just a 3x3 pixel space when the human retina has millions of photoreceptors.  This is !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

For example, say you are looking at a 3x3 area of red pixels, but 1 pixel slightly less red.  The human brain is unable to distinguish the slightly off-color pixel from the others.  Now if the off-color pixel were to keep losing intensity, eventually the human brain would be able to see a dark-red pixel standing apart from the other red pixels.  The human brain therefore has a certain level of pattern recognition sensitivity because it is limited by the number of neurons it has to recognize patterns.  

[GRAPHIC?]

However, the neocortex does not operate with colors, sounds, or tastes.  Fundamentally, a pattern in the neocortex is a set of neurons that activate in responce to an input at a certain time, or a spatial input.  For example, let's say you see a blue circle.  A set of neurons in your neocortex will activate in response to the visual stimulus of seeing the blue circle.  If the circle were suddenly to become red, another set of neurons would respond.  A sequence in the neocortex is therefore a sequential progression of neuron activations.  By converting, or encoding, observed patterns into neuron activations, the neocortex has a unified standard for storing and operating on knowledge.

![alt tag](https://raw.githubusercontent.com/ddigiorg/neuroowl.github.io/master/images/technology/divination_machine/knowledge_figure_2.png)

In the above graphic, pink and cyan circles as well as "wait" and "eat!" actions have been encoded into neuron activations.

## Code Functionality

DM is coded in C++ using OpenCL parallel processing framework.  Working operating systems include Linux and Windows.  The only dependancy needed to run DM is an OpenCL installation.

Divination Machine has 4 core functions:
- Spatial Encoding
- Temporal Encoding
- Spatial Decoding
- Learning

The figure below shows an example of a single step through DM.  Although its operation is quite involved, we will step through each function in its logical order and explain its functionality.

![alt tag](https://raw.githubusercontent.com/ddigiorg/neuroowl.github.io/master/images/technology/divination_machine/map.png)

### Spatial Encoding

Spatial Encoding observes input at a single time step and attempts to recognize a learned pattern.  The algorithm uses neuron activations to group together similar patterns from the near infinite possibilites of an input space.  

#### 1. setNeuronSums

Programatically, each column essentially behaves like a Self Organizing Map.  For every neuron, compute the Euclidian distance formula between its memories and its column's receptive field input values.  The Euclidian distance formula compares two sets of values and computes how similar they are to each other.  A small distance means the values are similar while large distance means the values are not similar.  Therefore, the output of this fuction is an array of floats indicating how well the neuron memories match with the input data.

The Euclidian Distance Formula:
for every memory m: sum += (input[m] - memory[m])<sup>2</sup>

For example:

sum =  
(0.1 - 0.1)<sup>2</sup> + (0.1 - 0.2)<sup>2</sup> + (0.1 - 0.3)<sup>2</sup> + (0.1 - 0.1)<sup>2</sup> + (0.1 - 0.2)<sup>2</sup> + (0.9 - 0.8)<sup>2</sup> + (0.1 - 0.1)<sup>2</sup> + (0.9 - 0.7)<sup>2</sup> + (0.9 - 0.9)<sup>2</sup>

sum = 0.11

#### 2. getMinIndices

For each column, the neuron with the smallest distance is the column's winner neuron, representing the input context of the column's receptive field input.  Thus the Spatial Encoder converts the entire input into a Sparse Distributed Representation (SDR) of neuron activations called Column Winners.  The minumum index of each column are highlighted in Neuron Sums.

#### 3. setPatternSums

Each row in Pattern Memories contains previously learned column winner SDRs.  The algorithm compares the values of Column Winners with each row in Pattern Memories.  For every equivalent value(representing the neuron index of a column), the respective sum is incremented by 1.

#### 4. getMatchingIndex

For every Pattern Sum index if the value is equal to the number of columns, then set Pattern to the Pattern Sum index.

### Temporal Encoding (Prediction)

Temporal Encoding observes sequences of patterns from Spatial Encoding and attempts to recognize a learned sequence.  If it recognizes a sequence, the algorithm outputs a prediction of the next time step's pattern.  By feeding this predicted pattern back through another  iteration of Temporal Encoding, the algorithm outputs a prediction of a pattern two time-steps ahead of the current time-step.  By continuing to feedback predicted patterns, DM can predict as many time-steps into the future as desired.

#### 5. setSTM

STM stands for Short Term Memory, which holds a sequence of patterns observed over time from Spatial Encoding.  The indices of STM represent historical time steps, ie. index 0 is t = 0, index 1 is t = -1, index 2 is t = -2 and so on.  First, the algorithm shifts pattern values in STM back through time, ie. the pattern in index 0 moves to index 1, the pattern in index 1 moves to index 2, and so on.  Then the new pattern is copied into index 0.  For example:

[INSERT GRAPHIC HERE]

What new pattern gets copied depends on what iteration of Temporal Encoding the algorithm is on.  For the first iteration the observed pattern from Spatial Encoding is placed into index 0 of STM.  If there are subsequent iterations of Temporal Memory, the current values of STM are copied into a temporary buffer.  As Temporal Encoding iterates, STM shifts and predicted patterns are placed into index 0 of STM.  When the last iteration of Temporal Memory is finished, the values of the temporary STM are copied back into STM.  This allows DM to keep observed and predicted sequences seperate.

#### 6. setSequenceSums

Each row in Sequence Memories contains previously learned Short Term Memories.  The algorithm compares the values of STM(index 0 to maxIndex - 1) with each row in Sequence Memories.  For every equivalent value, the respective sum is incremented by 1.

#### 7. getMatchingIndex

For every Sequence Sum index if the value is equal to the number of patterns in Short Term Memory minus 1, then set Sequence to the Sequence Sum index.

#### 8. getPredictPattern

Retrieve the Predict Pattern value from Predict Memories by indexing off the value of Sequence.

### Decoding

Decoding uses memories to convert neuron patterns into output data.  Decoding can be thought of as DM "communicating" possible future patterns.

#### 9. getOutputs

Once Temporal Memory has completed its predicting and have a value in Predict Pattern, the algorithm grabs the appropriate column winner SDR from Pattern Memories.  Using this set of column winners, the algorithm looks up each winner neuron's memories and applies the memories to the output receptive field of each column.

### Learning

#### 10. learnNeurons

#### 11. learnPattern

If Column Winners values are not in Pattern Memories, add those values to Pattern Memories.

#### 12. learnSequence

If Short Term Memory values at index 1 and above are not in Sequence Memories, add those values to Sequence Memories.  Additionally, the Short Term Memory value at index 0 is added Predict Memories at the same row location of the new sequence in Sequence Memories.

## Possible Future Improvements
- Convert OpenCL images to buffers for much larger memory
- Fix learnInputMemories by having each neuron learn to recognize different field values



## To-Do list
- Finish README
- Implement a "forecast" function to show running future predictions
- Improve pattern and sequence recognition by chunking setPatternSums and setSequenceSums
- Show a very simple moving square demo
