<img src="https://raw.githubusercontent.com/ddigiorg/neuroowl.github.io/master/images/technology/divination_machine/logo.png" alt="DM Logo" width=100/>

# Divination Machine

Divination Machine (DM) is a Machine Intelligence architecture based on intelligence principles of the mammalian neocortex and is inspired by research from [Numenta's](http://numenta.com/) [Hierarchical Temporal Memory (HTM)](https://github.com/numenta/nupic) and [Ogma's](https://ogma.ai/) [Feynman Machine (FM)](https://github.com/ogmacorp/OgmaNeo).  Although DM's name implies the invocation of the supernatural, its functionality is obviously earthly.  Divination Machine at each time step:

- Observes spatial-temporal sensory-motor input
- Encodes inputs into a sparsly distributed neuron activation pattern
- Predicts the most likely future pattern
- Decodes future pattern into visible spatial-temporal sensory-motor outputs
- Learns new neuron activation patterns and sequences of patterns

## Code Functionality

DM is coded in C++ using the OpenCL parallel processing framework.  Working operating systems include Linux and Windows.  The only dependancy needed to run DM is an OpenCL installation.

Divination Machine has 4 core functions explained in this paper:
- Spatial Encoding
- Temporal Encoding
- Spatial Decoding
- Learning

The figure below shows an example of a single time-step through DM.  Although its operation is quite involved, this paper explains each function in the order they occour.

![alt tag](https://raw.githubusercontent.com/ddigiorg/neuroowl.github.io/master/images/technology/divination_machine/map.png)

### Spatial Encoding

Spatial Encoding observes input at a single time-step and converts it into a Sparse Distributed Eepresentation (SDR) representing neuron activations.  The algorithm then tries to match the SDR to a bank of SDRs called "Pattern Memories".  If a match is found, the index of the matching SDR in the bank is the "Pattern Memory".  If no match is found, learning occours.

#### 1. setNeuronSums

Programatically each column behaves like a Self Organizing Map, an unsupervised learning technique used for dimensionality reduction(reducing the near infinite possibilites of an input space into a limited set of descrete activations).  The column is centered at a specified location on the input space and observes a receptive field, a specified size of the input space.  

Each column tries calculate the best matching neuron by calculating the Euclidian distance formula between the neuron's memories and the column's observed receptive field values.  The Euclidian distance indicates the level of similarity between two sets of values.  A short distance means the values are similar while a long distance means the values are different.  

The Euclidian Distance Formula:
```
for every memory m: sum += (input[m] - memory[m])<sup>2</sup>
```

Therefore, the output of this fuction is a 2D array of floats called "Neuron Sums" representing how well each column's neuron memories match with the input data of the column's respective receptive field.

#### 2. getMinIndices

For each column, the neuron with the smallest distance is the best matching neuron and it's integer index gets written to "Column Winners".  Thus the output of this function is a vector of integers representing a SDR of column neuron activations.

#### 3. setPatternSums

The algorithm stores SDRs in a bank called "Pattern Memories" where each row contains previously learned "Column Winners".  At each time step the algorithm compares "Column Winners" with each row in "Pattern Memories".  Each equivalent value increments the row's sum by 1.

#### 4. getMatchingIndex

This function looks at each row of "Pattern Sums" and if the sum is equal to the number of columns, the row index is stored in "Pattern".  This stored value represents the row index of "Pattern Memories" that matches "Column Winners".  If no rows match, the next available empty row index value is used.

### Temporal Encoding (Prediction)

Temporal Encoding observes sequences of patterns from Spatial Encoding and attempts to recognize a learned sequence.  If it recognizes a sequence, the algorithm outputs a prediction of the next time step's pattern.  By feeding this predicted pattern back through another  iteration of Temporal Encoding, the algorithm outputs a prediction of a pattern two time-steps ahead of the current time-step.  By continuing to feedback predicted patterns, DM can predict as many time-steps into the future as desired.

Prediction is one or more Temporal Encoding steps.

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

### Spatial Decoding

Decoding uses memories to convert neuron patterns into output data.  Decoding can be thought of as DM "communicating" possible future patterns.

#### 9. getOutputs

Once Temporal Memory has completed its predicting and have a value in Predict Pattern, the algorithm grabs the appropriate column winner SDR from Pattern Memories.  Using this set of column winners, the algorithm looks up each winner neuron's memories and applies the memories to the output receptive field of each column.

### Learning

#### 10. learnNeurons

#### 11. learnPattern

If Column Winners values are not in Pattern Memories, add those values to Pattern Memories.

#### 12. learnSequence

If Short Term Memory values at index 1 and above are not in Sequence Memories, add those values to Sequence Memories.  Additionally, the Short Term Memory value at index 0 is added Predict Memories at the same row location of the new sequence in Sequence Memories.
