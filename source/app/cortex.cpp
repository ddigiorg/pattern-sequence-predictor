// ==========
// cortex.cpp
// ==========

#include "cortex.h"

Cortex::Cortex(std::mt19937 rng)
{
	_rng = rng;

	_zeroOrigin[0] = 0;
	_zeroOrigin[1] = 0;
	_zeroOrigin[2] = 0;
}

void Cortex::initialize(ComputeSystem &cs, ComputeProgram &cp)
{
	// Initialize kernels
	_initWeightsKernel = cl::Kernel(cp.getProgram(), "initWeights");
	_setValuesFromWinnersKernel = cl::Kernel(cp.getProgram(), "setValuesFromWinners");
	_setSumsFromValuesKernel = cl::Kernel(cp.getProgram(), "setSumsFromValues");
	_setWinnersFromSumsKernel = cl::Kernel(cp.getProgram(), "setWinnersFromSums");

	_setSumsFromWinnersKernel = cl::Kernel(cp.getProgram(), "setSumsFromWinners");
	_setVotesFromSumKernel = cl::Kernel(cp.getProgram(), "setVotesFromSum");

	_setValuesFromPredictsKernel = cl::Kernel(cp.getProgram(), "setValuesFromPredicts");
	_learnWeightsFromPreviousValuesKernel = cl::Kernel(cp.getProgram(), "learnWeightsFromPreviousValues");

	// Define images
	_visibleBlock.inputs  = utils::createImage2D(cs, _visibleBlock.visibleSize, CL_R, CL_FLOAT);
	_visibleBlock.outputs = utils::createImage2D(cs, _visibleBlock.visibleSize, CL_R, CL_FLOAT);
	_visibleBlock.sums    = utils::createImage2D(cs, _visibleBlock.blockSize,   CL_R, CL_FLOAT);
	_visibleBlock.weights = utils::createImage3D(cs, _visibleBlock.weightSize,  CL_R, CL_FLOAT);

	_memoryBlock.chunkWinners = utils::createImage2D(cs, _memoryBlock.numChunks,  CL_RG, CL_FLOAT);
	_memoryBlock.sums         = utils::createImage2D(cs, _memoryBlock.blockSize,  CL_R,  CL_FLOAT);
	_memoryBlock.weights      = utils::createImage3D(cs, _memoryBlock.weightSize, CL_R,  CL_FLOAT);

	// Fill images with zeros
	cs.getQueue().enqueueFillImage(_visibleBlock.inputs,  _zeroColor, _zeroOrigin, _visibleBlock.clVisibleRegion);
	cs.getQueue().enqueueFillImage(_visibleBlock.outputs, _zeroColor, _zeroOrigin, _visibleBlock.clVisibleRegion);
	cs.getQueue().enqueueFillImage(_visibleBlock.sums,    _zeroColor, _zeroOrigin, _visibleBlock.clBlockRegion  );

	cs.getQueue().enqueueFillImage(_memoryBlock.chunkWinners, _zeroColor, _zeroOrigin, _memoryBlock.clChunkRegion );
	cs.getQueue().enqueueFillImage(_memoryBlock.sums,         _zeroColor, _zeroOrigin, _memoryBlock.clBlockRegion );

	cs.getQueue().finish();

	// Fill 3D images with random float values
	initWeights(cs, _visibleBlock.weights, _visibleBlock.weightSize, _visibleBlock.initWeightRange);

	initWeights(cs, _memoryBlock.weights, _memoryBlock.weightSize, _memoryBlock.initWeightRange);

}

void Cortex::step(ComputeSystem& cs, bool learn)
{
	cl::NDRange range;

	// Encode
	_setSumsFromValuesKernel.setArg(0, _visibleBlock.inputs);
	_setSumsFromValuesKernel.setArg(1, _visibleBlock.weights);
	_setSumsFromValuesKernel.setArg(2, _visibleBlock.sums);
	_setSumsFromValuesKernel.setArg(3, _visibleBlock.chunkSize);
	_setSumsFromValuesKernel.setArg(4, _visibleBlock.visibleSize);
	_setSumsFromValuesKernel.setArg(5, _visibleBlock.fieldSize);
	_setSumsFromValuesKernel.setArg(6, _visibleBlock.fieldStart);
	_setSumsFromValuesKernel.setArg(7, _visibleBlock.fieldStop);
	_setSumsFromValuesKernel.setArg(8, _visibleBlock.fieldOffset);

	range = cl::NDRange(_visibleBlock.blockSize.x, _visibleBlock.blockSize.y);
	cs.getQueue().enqueueNDRangeKernel(_setSumsFromValuesKernel, cl::NullRange, range);
	cs.getQueue().finish();

	_setWinnersFromSumsKernel.setArg(0, _visibleBlock.sums);
	_setWinnersFromSumsKernel.setArg(1, _memoryBlock.chunkWinners);
	_setWinnersFromSumsKernel.setArg(2, _visibleBlock.chunkSize);

	range = cl::NDRange(_visibleBlock.numChunks.x, _visibleBlock.numChunks.y);
	cs.getQueue().enqueueNDRangeKernel(_setWinnersFromSumsKernel, cl::NullRange, range);
	cs.getQueue().finish();

	// Predict
	_setSumsFromWinnersKernel.setArg(0, _memoryBlock.chunkWinners);
	_setSumsFromWinnersKernel.setArg(1, _memoryBlock.weights);
	_setSumsFromWinnersKernel.setArg(2, _memoryBlock.sums);
	_setSumsFromWinnersKernel.setArg(3, _memoryBlock.chunkSize);
	_setSumsFromWinnersKernel.setArg(4, _memoryBlock.hiddenSize);
	_setSumsFromWinnersKernel.setArg(5, _memoryBlock.fieldSize);
	_setSumsFromWinnersKernel.setArg(6, _memoryBlock.fieldStart);
	_setSumsFromWinnersKernel.setArg(7, _memoryBlock.fieldStop);
	_setSumsFromWinnersKernel.setArg(8, _memoryBlock.fieldOffset);

	range = cl::NDRange(_memoryBlock.blockSize.x, _memoryBlock.blockSize.y);
	cs.getQueue().enqueueNDRangeKernel(_setSumsFromWinnersKernel, cl::NullRange, range);
	cs.getQueue().finish();

	// !!!!!!!!!!!!!!!!!!!STOPPED HERE!!!!!!!!!!!!!!!!!!!!!!!

	// Decode
	_setValuesFromPredictsKernel.setArg(0, _memoryBlock.chunkWinners);
	_setValuesFromPredictsKernel.setArg(1, _visibleBlock.weights);
	_setValuesFromPredictsKernel.setArg(2, _visibleBlock.outputs);
	_setValuesFromPredictsKernel.setArg(3, _visibleBlock.chunkSize);
	_setValuesFromPredictsKernel.setArg(4, _visibleBlock.visibleSize);
	_setValuesFromPredictsKernel.setArg(5, _visibleBlock.fieldSize);
	_setValuesFromPredictsKernel.setArg(6, _visibleBlock.fieldStart);
	_setValuesFromPredictsKernel.setArg(7, _visibleBlock.fieldStop);
	_setValuesFromPredictsKernel.setArg(8, _visibleBlock.fieldOffset);

	range = cl::NDRange(_visibleBlock.numChunks.x, _visibleBlock.numChunks.y);
	cs.getQueue().enqueueNDRangeKernel(_setValuesFromPredictsKernel, cl::NullRange, range);
	cs.getQueue().finish();

	// Learn
	if (learn)
	{
		_learnWeightsFromPreviousValuesKernel.setArg( 0, _visibleBlock.inputs);
		_learnWeightsFromPreviousValuesKernel.setArg( 1, _memoryBlock.chunkWinners);
		_learnWeightsFromPreviousValuesKernel.setArg( 2, _visibleBlock.weights);
		_learnWeightsFromPreviousValuesKernel.setArg( 3, _visibleBlock.weights);
		_learnWeightsFromPreviousValuesKernel.setArg( 4, _visibleBlock.chunkSize);
		_learnWeightsFromPreviousValuesKernel.setArg( 5, _visibleBlock.visibleSize);
		_learnWeightsFromPreviousValuesKernel.setArg( 6, _visibleBlock.fieldSize);
		_learnWeightsFromPreviousValuesKernel.setArg( 7, _visibleBlock.fieldStart);
		_learnWeightsFromPreviousValuesKernel.setArg( 8, _visibleBlock.fieldStop);
		_learnWeightsFromPreviousValuesKernel.setArg( 9, _visibleBlock.fieldOffset);
		_learnWeightsFromPreviousValuesKernel.setArg(10, _visibleBlock.learningRate);

		range = cl::NDRange(_visibleBlock.numChunks.x, _visibleBlock.numChunks.y);
		cs.getQueue().enqueueNDRangeKernel(_learnWeightsFromPreviousValuesKernel, cl::NullRange, range);
		cs.getQueue().finish();
	}

	// Swap
}

void Cortex::setVisibleInputs(ComputeSystem& cs, std::vector<float> dataVector)
{
	cs.getQueue().enqueueWriteImage(
		_visibleBlock.inputs, CL_TRUE, _zeroOrigin, _visibleBlock.clVisibleRegion, 0, 0, dataVector.data());
	cs.getQueue().finish();
}


std::vector<float> Cortex::getVisibleInputs(ComputeSystem &cs)
{
	std::vector<float> dataVector(_visibleBlock.visibleSize.x * _visibleBlock.visibleSize.y);

	cs.getQueue().enqueueReadImage(
		_visibleBlock.inputs, CL_TRUE, _zeroOrigin, _visibleBlock.clVisibleRegion, 0, 0, dataVector.data());
	cs.getQueue().finish();

	return dataVector;
}

std::vector<float> Cortex::getVisibleOutputs(ComputeSystem &cs)
{
	std::vector<float> dataVector(_visibleBlock.visibleSize.x * _visibleBlock.visibleSize.y);

	cs.getQueue().enqueueReadImage(
		_visibleBlock.outputs, CL_TRUE, _zeroOrigin, _visibleBlock.clVisibleRegion, 0, 0, dataVector.data());
	cs.getQueue().finish();

	return dataVector;
}

std::vector<float> Cortex::getChunkSDR(ComputeSystem &cs, unsigned int memoryIndex)
{
	cl::Image2D sdr = utils::createImage2D(cs, _memoryBlock.blockSize, CL_R, CL_FLOAT);

	_setValuesFromWinnersKernel.setArg(0, _memoryBlock.chunkWinners);
	_setValuesFromWinnersKernel.setArg(1, sdr);
	_setValuesFromWinnersKernel.setArg(2, _memoryBlock.chunkSize);

	cl::NDRange range = cl::NDRange(_memoryBlock.numChunks.x, _memoryBlock.numChunks.y);
	cs.getQueue().enqueueNDRangeKernel(_setValuesFromWinnersKernel, cl::NullRange, range);
	cs.getQueue().finish();

	std::vector<float> dataVector(_memoryBlock.blockSize.x * _memoryBlock.blockSize.y);

	cs.getQueue().enqueueReadImage(
		sdr, CL_TRUE, _zeroOrigin, _memoryBlock.clBlockRegion, 0, 0, dataVector.data());
	cs.getQueue().finish();

	return dataVector;
}


void Cortex::initWeights(ComputeSystem &cs, cl::Image3D weights, cl_int3 weightSize, cl_float2 weightRange)
{
	std::uniform_int_distribution<int> seedDist(0, 999);

	cl_uint2 seed = {(cl_uint)seedDist(_rng), (cl_uint)seedDist(_rng)};

	_initWeightsKernel.setArg(0, weights);
	_initWeightsKernel.setArg(1, weightRange);
	_initWeightsKernel.setArg(2, seed);

	cs.getQueue().enqueueNDRangeKernel(_initWeightsKernel, cl::NullRange, cl::NDRange(weightSize.x, weightSize.y, weightSize.z));
	cs.getQueue().finish();
}
