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

	void addMemoryBlocks(
		unsigned int numBlocks,
		utils::Vec2ui32 blockSize,
		utils::Vec2ui32 chunkSize,
		float learningRate)
	{
//		MemoryBlock *mB = new MemoryBlock;
//		mB->initialize(blockSize, chunkSize, learningRate);
//		_memoryBlocks.push_back(*mB);

		_memoryBlocks.resize(numBlocks + 1);

		for (int mb = 0; mb < _memoryBlocks.size(); mb++)
		{
			_memoryBlocks[mb].initialize(blockSize, chunkSize, learningRate); 
		}
	}

	void addPredictBlock(
		utils::Vec2ui32 blockSize,
		utils::Vec2ui32 chunkSize)
	{
		_predictBlock.initialize(blockSize, chunkSize);
	}

	void initialize(ComputeSystem &cs, ComputeProgram &cp);

	void step(ComputeSystem& cs, bool learn);

	void setVisibleInputs(ComputeSystem &cs, std::vector<float> dataVector);

	std::vector<float> getVisibleInputs(ComputeSystem &cs);
	std::vector<float> getVisibleOutputs(ComputeSystem &cs);
	std::vector<float> getChunkWinnersOldest(ComputeSystem &cs);
	std::vector<float> getChunkWinners(ComputeSystem &cs, unsigned int memoryIndex);
	std::vector<float> getChunkPredicts(ComputeSystem &cs);


private:
	std::mt19937 _rng;

	cl_float4 _zeroColor = {0.0f, 0.0f, 0.0f, 0.0f};

	cl::size_t<3> _zeroOrigin;

	VisibleBlock _visibleBlock;

	std::vector<MemoryBlock> _memoryBlocks;

	PredictBlock _predictBlock;

	cl::Kernel _initWeightsKernel;
	cl::Kernel _setValuesFromWinnersKernel;

	// Encode
	cl::Kernel _setSumsFromVisiblesKernel;
	cl::Kernel _setWinnersFromSumsKernel;

	// Predict
	cl::Kernel _setSumsFromWinnersKernel;
	cl::Kernel _setVotesFromSumsKernel;
	cl::Kernel _setPredictsFromVotesKernel;

	// Decode
	cl::Kernel _setVisiblesFromPredictsKernel;

	// Learn
	cl::Kernel _learnWeightsFromPreviousVisiblesKernel;
	cl::Kernel _learnWeightsFromPreviousWinnersKernel;

	// Swap

	void initWeights(ComputeSystem &cs, cl::Image3D weights, cl_int3 weightSize, cl_float2 weightRange);
};

#endif
