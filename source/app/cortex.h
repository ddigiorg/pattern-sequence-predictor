// ========
// cortex.h
// ========

#ifndef CORTEX_H
#define CORTEX_H

#include "utils/utils.h"

#include "compute/compute-system.h"
#include "compute/compute-program.h"

#include "visible_block.h"
#include "memory_block.h"
#include "predict_block.h"

#include <vector>
#include <random>
#include <iostream>

class Cortex
{
public:
	Cortex(std::mt19937 rng);

	void addVisibleBlock(
		utils::Vec2ui32 blockSize,
		utils::Vec2ui32 chunkSize,
		utils::Vec2ui32 visibleSize,
		utils::Vec2ui32 fieldSize,
		float learningRate)
	{
		_visibleBlock.initialize(blockSize, chunkSize, visibleSize, fieldSize, learningRate);
	}

	void addMemoryBlock(
		utils::Vec2ui32 blockSize,
		utils::Vec2ui32 chunkSize,
		utils::Vec2ui32 hiddenSize,
		utils::Vec2ui32 fieldSize,
		float learningRate)
	{
		_memoryBlock.initialize(blockSize, chunkSize, hiddenSize, fieldSize, learningRate);
	}

	void addPredictBlock(
		utils::Vec2ui32 blockSize,
		utils::Vec2ui32 chunkSize,
		utils::Vec2ui32 hiddenSize,
		utils::Vec2ui32 fieldSize)
	{
		_predictBlock.initialize(blockSize, chunkSize, hiddenSize, fieldSize);
	}

	void initialize(ComputeSystem &cs, ComputeProgram &cp);

	void step(ComputeSystem& cs, bool learn);

	void setVisibleInputs(ComputeSystem &cs, std::vector<float> dataVector);

	std::vector<float> getVisibleInputs(ComputeSystem &cs);
	std::vector<float> getVisibleOutputs(ComputeSystem &cs);
	std::vector<float> getChunkSDR(ComputeSystem &cs, unsigned int memoryIndex);

private:
	std::mt19937 _rng;

	cl_float4 _zeroColor = {0.0f, 0.0f, 0.0f, 0.0f};

	cl::size_t<3> _zeroOrigin;

	VisibleBlock _visibleBlock;

	MemoryBlock _memoryBlock;

	PredictBlock _predictBlock;

	cl::Kernel _initWeightsKernel;
	cl::Kernel _setValuesFromWinnersKernel;

	// Encode
	cl::Kernel _setSumsFromValuesKernel;
	cl::Kernel _setWinnersFromSumsKernel;

	// Predict
	cl::Kernel _setSumsFromWinnersKernel;
	cl::Kernel _setVotesFromSumKernel;
	cl::Kernel _setPredictsFromVotes;

	// Decode
	cl::Kernel _setValuesFromPredictsKernel;

	// Learn
	cl::Kernel _learnWeightsFromPreviousValuesKernel;

	// Swap

	void initWeights(ComputeSystem &cs, cl::Image3D weights, cl_int3 weightSize, cl_float2 weightRange);
};

#endif
