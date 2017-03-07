# Divination Machine

- Each column looks at a receptive field in the input data space and encodes repeatedly seen patterns into a single neuron activation, or "winner neuron".
- The addresses of winner neurons for every column are stored in pattern memories.  The index of the pattern is used for sequence memory.
- specific sequences of patterns are stored in sequence memory and predict memory.

## To-Do list
- Finish README
- Implement a "forecast" function to show running future predictions
- Show a very simple moving square demo

## Improvements
- Improve pattern and sequence recognition by chunking setPatternSums and setSequenceSums
- Change OpenCL images to buffers for much larger space allocation
- Fix learnInputMemories by having each neuron learn to recognize different field values
- Once learnInputMemories fixed, make 2d ball physics demo

test
