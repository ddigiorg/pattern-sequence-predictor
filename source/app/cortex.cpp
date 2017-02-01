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

	_setSumsFromVisiblesKernel = cl::Kernel(cp.getProgram(), "setSumsFromVisibles");
	_setWinnersFromSumsKernel = cl::Kernel(cp.getProgram(), "setWinnersFromSums");

	_setSumsFromWinnersKernel = cl::Kernel(cp.getProgram(), "setSumsFromWinners");
	_setVotesFromSumsKernel = cl::Kernel(cp.getProgram(), "setVotesFromSums");
	_setPredictsFromVotesKernel = cl::Kernel(cp.getProgram(), "setPredictsFromVotes");

	_setVisiblesFromPredictsKernel = cl::Kernel(cp.getProgram(), "setVisiblesFromPredicts");

	_learnWeightsFromPreviousVisiblesKernel = cl::Kernel(cp.getProgram(), "learnWeightsFromPreviousVisibles");
	_learnWeightsFromPreviousWinnersKernel = cl::Kernel(cp.getProgram(), "learnWeightsFromPreviousWinners");

	// Initialize _visibleBlock
	_visibleBlock.inputs  = utils::createImage2D(cs, _visibleBlock.visibleSize, CL_R, CL_FLOAT);
	_visibleBlock.outputs = utils::createImage2D(cs, _visibleBlock.visibleSize, CL_R, CL_FLOAT);
	_visibleBlock.sums    = utils::createImage2D(cs, _visibleBlock.blockSize,   CL_R, CL_FLOAT);
	_visibleBlock.weights = utils::createImage3D(cs, _visibleBlock.weightSize,  CL_R, CL_FLOAT);

	cs.getQueue().enqueueFillImage(_visibleBlock.inputs,  _zeroColor, _zeroOrigin, _visibleBlock.clVisibleRegion);
	cs.getQueue().enqueueFillImage(_visibleBlock.outputs, _zeroColor, _zeroOrigin, _visibleBlock.clVisibleRegion);
	cs.getQueue().enqueueFillImage(_visibleBlock.sums,    _zeroColor, _zeroOrigin, _visibleBlock.clBlockRegion  );

	initWeights(cs, _visibleBlock.weights, _visibleBlock.weightSize, _visibleBlock.initWeightRange);

	// Initialize _memoryBlocks
	int mb = _memoryBlocks.size() - 1;

	_memoryBlocks[mb].chunkWinners = utils::createImage2D(cs, _memoryBlocks[mb].numChunks,  CL_RG, CL_FLOAT);

	cs.getQueue().enqueueFillImage(_memoryBlocks[mb].chunkWinners, _zeroColor, _zeroOrigin, _memoryBlocks[mb].clChunkRegion );

	for (int mb = _memoryBlocks.size() - 2; mb >= 0; mb--)
	{
		_memoryBlocks[mb].chunkWinners = utils::createImage2D(cs, _memoryBlocks[mb].numChunks,  CL_RG, CL_FLOAT);
		_memoryBlocks[mb].sums         = utils::createImage2D(cs, _memoryBlocks[mb].blockSize,  CL_R,  CL_FLOAT);
		_memoryBlocks[mb].weights      = utils::createImage3D(cs, _memoryBlocks[mb].weightSize, CL_R,  CL_FLOAT);

		cs.getQueue().enqueueFillImage(_memoryBlocks[mb].chunkWinners, _zeroColor, _zeroOrigin, _memoryBlocks[mb].clChunkRegion );
		cs.getQueue().enqueueFillImage(_memoryBlocks[mb].sums,         _zeroColor, _zeroOrigin, _memoryBlocks[mb].clBlockRegion );

		initWeights(cs, _memoryBlocks[mb].weights, _memoryBlocks[mb].weightSize, _memoryBlocks[mb].initWeightRange);
	}

	// Initialize _predictBlock
	_predictBlock.votes         = utils::createImage2D(cs, _predictBlock.blockSize, CL_R,  CL_FLOAT);
	_predictBlock.chunkPredicts = utils::createImage2D(cs, _predictBlock.numChunks, CL_RG, CL_FLOAT);

	cs.getQueue().enqueueFillImage(_predictBlock.votes,         _zeroColor, _zeroOrigin, _predictBlock.clBlockRegion);
	cs.getQueue().enqueueFillImage(_predictBlock.chunkPredicts, _zeroColor, _zeroOrigin, _predictBlock.clChunkRegion);

	cs.getQueue().finish();
}

void Cortex::step(ComputeSystem& cs, bool learn)
{
	cl::NDRange range;

	// Swap
	for (int mb = _memoryBlocks.size() - 2; mb >= 0; mb--)
	{
		cs.getQueue().enqueueCopyImage(
			_memoryBlocks[mb].chunkWinners, _memoryBlocks[mb + 1].chunkWinners,
			_zeroOrigin, _zeroOrigin, _memoryBlocks[mb].clChunkRegion);
	}

	cs.getQueue().enqueueFillImage(_predictBlock.votes, _zeroColor, _zeroOrigin, _predictBlock.clBlockRegion);
	cs.getQueue().finish();

	// Encode
	_setSumsFromVisiblesKernel.setArg(0, _visibleBlock.inputs);
	_setSumsFromVisiblesKernel.setArg(1, _visibleBlock.weights);
	_setSumsFromVisiblesKernel.setArg(2, _visibleBlock.sums);
	_setSumsFromVisiblesKernel.setArg(3, _visibleBlock.chunkSize);
	_setSumsFromVisiblesKernel.setArg(4, _visibleBlock.visibleSize);
	_setSumsFromVisiblesKernel.setArg(5, _visibleBlock.fieldSize);
	_setSumsFromVisiblesKernel.setArg(6, _visibleBlock.fieldOffset);

	range = cl::NDRange(_visibleBlock.blockSize.x, _visibleBlock.blockSize.y);
	cs.getQueue().enqueueNDRangeKernel(_setSumsFromVisiblesKernel, cl::NullRange, range);
	cs.getQueue().finish();

	_setWinnersFromSumsKernel.setArg(0, _visibleBlock.sums);
	_setWinnersFromSumsKernel.setArg(1, _memoryBlocks[0].chunkWinners);
	_setWinnersFromSumsKernel.setArg(2, _visibleBlock.chunkSize);

	range = cl::NDRange(_visibleBlock.numChunks.x, _visibleBlock.numChunks.y);
	cs.getQueue().enqueueNDRangeKernel(_setWinnersFromSumsKernel, cl::NullRange, range);
	cs.getQueue().finish();

	// Predict
	/*
	_setWinnersFromSumsKernel.setArg(0, _memoryBlocks[0].sums);
	_setWinnersFromSumsKernel.setArg(1, _predictBlock.chunkPredicts);
	_setWinnersFromSumsKernel.setArg(2, _predictBlock.chunkSize);

	range = cl::NDRange(_predictBlock.numChunks.x, _predictBlock.numChunks.y);
	cs.getQueue().enqueueNDRangeKernel(_setWinnersFromSumsKernel, cl::NullRange, range);
	cs.getQueue().finish();
	*/

	for (int mb = _memoryBlocks.size() - 2; mb >= 0; mb--)
	{
		_setSumsFromWinnersKernel.setArg(0, _memoryBlocks[mb].chunkWinners);
		_setSumsFromWinnersKernel.setArg(1, _memoryBlocks[mb].weights);
		_setSumsFromWinnersKernel.setArg(2, _memoryBlocks[mb].sums);
		_setSumsFromWinnersKernel.setArg(3, _memoryBlocks[mb].chunkSize);

		range = cl::NDRange(_memoryBlocks[mb].blockSize.x, _memoryBlocks[mb].blockSize.y);
		cs.getQueue().enqueueNDRangeKernel(_setSumsFromWinnersKernel, cl::NullRange, range);
		cs.getQueue().finish();

		_setVotesFromSumsKernel.setArg(0, _memoryBlocks[mb].sums);
		_setVotesFromSumsKernel.setArg(1, _predictBlock.votes);
		_setVotesFromSumsKernel.setArg(2, _predictBlock.votes);
		_setVotesFromSumsKernel.setArg(3, _predictBlock.chunkSize);

		range = cl::NDRange(_predictBlock.numChunks.x, _predictBlock.numChunks.y);
		cs.getQueue().enqueueNDRangeKernel(_setVotesFromSumsKernel, cl::NullRange, range);
		cs.getQueue().finish();
	}

	_setPredictsFromVotesKernel.setArg(0, _predictBlock.votes);
	_setPredictsFromVotesKernel.setArg(1, _predictBlock.chunkPredicts);
	_setPredictsFromVotesKernel.setArg(2, _predictBlock.chunkSize);

	range = cl::NDRange(_predictBlock.numChunks.x, _predictBlock.numChunks.y);
	cs.getQueue().enqueueNDRangeKernel(_setPredictsFromVotesKernel, cl::NullRange, range);
	cs.getQueue().finish();

	// Decode
	_setVisiblesFromPredictsKernel.setArg(0, _predictBlock.chunkPredicts);
	_setVisiblesFromPredictsKernel.setArg(1, _visibleBlock.weights);
	_setVisiblesFromPredictsKernel.setArg(2, _visibleBlock.outputs);
	_setVisiblesFromPredictsKernel.setArg(3, _predictBlock.chunkSize);
	_setVisiblesFromPredictsKernel.setArg(4, _visibleBlock.visibleSize);
	_setVisiblesFromPredictsKernel.setArg(5, _visibleBlock.fieldSize);
	_setVisiblesFromPredictsKernel.setArg(6, _visibleBlock.fieldOffset);

	range = cl::NDRange(_visibleBlock.numChunks.x, _visibleBlock.numChunks.y);
	cs.getQueue().enqueueNDRangeKernel(_setVisiblesFromPredictsKernel, cl::NullRange, range);
	cs.getQueue().finish();

	// Learn
	if (learn)
	{
		_learnWeightsFromPreviousVisiblesKernel.setArg(0, _visibleBlock.inputs);
		_learnWeightsFromPreviousVisiblesKernel.setArg(1, _memoryBlocks[0].chunkWinners);  //memory block 0
		_learnWeightsFromPreviousVisiblesKernel.setArg(2, _visibleBlock.weights);
		_learnWeightsFromPreviousVisiblesKernel.setArg(3, _visibleBlock.weights);
		_learnWeightsFromPreviousVisiblesKernel.setArg(4, _visibleBlock.chunkSize);
		_learnWeightsFromPreviousVisiblesKernel.setArg(5, _visibleBlock.visibleSize);
		_learnWeightsFromPreviousVisiblesKernel.setArg(6, _visibleBlock.fieldSize);
		_learnWeightsFromPreviousVisiblesKernel.setArg(7, _visibleBlock.fieldOffset);
		_learnWeightsFromPreviousVisiblesKernel.setArg(8, _visibleBlock.learningRate);

		range = cl::NDRange(_visibleBlock.numChunks.x, _visibleBlock.numChunks.y);
		cs.getQueue().enqueueNDRangeKernel(_learnWeightsFromPreviousVisiblesKernel, cl::NullRange, range);
		cs.getQueue().finish();

		for (int mb = _memoryBlocks.size() - 2; mb >= 0; mb--)
		{
			_learnWeightsFromPreviousWinnersKernel.setArg(0, _memoryBlocks[mb + 1].chunkWinners);
			_learnWeightsFromPreviousWinnersKernel.setArg(1, _memoryBlocks[0].chunkWinners);  //memory block 0
			_learnWeightsFromPreviousWinnersKernel.setArg(2, _memoryBlocks[mb].weights);
			_learnWeightsFromPreviousWinnersKernel.setArg(3, _memoryBlocks[mb].weights);
			_learnWeightsFromPreviousWinnersKernel.setArg(4, _memoryBlocks[mb].chunkSize);
			_learnWeightsFromPreviousWinnersKernel.setArg(5, _memoryBlocks[mb].learningRate);

			range = cl::NDRange(_memoryBlocks[mb].numChunks.x, _memoryBlocks[mb].numChunks.y);
			cs.getQueue().enqueueNDRangeKernel(_learnWeightsFromPreviousWinnersKernel, cl::NullRange, range);
			cs.getQueue().finish();
		}
	}
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

std::vector<float> Cortex::getChunkWinnersOldest(ComputeSystem &cs)
{
	cl::Image2D sdr = utils::createImage2D(cs, _memoryBlocks[0].blockSize, CL_R, CL_FLOAT);

	int mb = _memoryBlocks.size() - 1;

	_setValuesFromWinnersKernel.setArg(0, _memoryBlocks[mb].chunkWinners);
	_setValuesFromWinnersKernel.setArg(1, sdr);
	_setValuesFromWinnersKernel.setArg(2, _memoryBlocks[mb].chunkSize);

	cl::NDRange range = cl::NDRange(_memoryBlocks[0].numChunks.x, _memoryBlocks[0].numChunks.y);
	cs.getQueue().enqueueNDRangeKernel(_setValuesFromWinnersKernel, cl::NullRange, range);
	cs.getQueue().finish();

	std::vector<float> dataVector(_memoryBlocks[0].blockSize.x * _memoryBlocks[0].blockSize.y);

	cs.getQueue().enqueueReadImage(
		sdr, CL_TRUE, _zeroOrigin, _memoryBlocks[0].clBlockRegion, 0, 0, dataVector.data());
	cs.getQueue().finish();

	return dataVector;
}


std::vector<float> Cortex::getChunkWinners(ComputeSystem &cs, unsigned int memoryIndex)
{
	cl::Image2D sdr = utils::createImage2D(cs, _memoryBlocks[0].blockSize, CL_R, CL_FLOAT);

	_setValuesFromWinnersKernel.setArg(0, _memoryBlocks[0].chunkWinners);
	_setValuesFromWinnersKernel.setArg(1, sdr);
	_setValuesFromWinnersKernel.setArg(2, _memoryBlocks[0].chunkSize);

	cl::NDRange range = cl::NDRange(_memoryBlocks[0].numChunks.x, _memoryBlocks[0].numChunks.y);
	cs.getQueue().enqueueNDRangeKernel(_setValuesFromWinnersKernel, cl::NullRange, range);
	cs.getQueue().finish();

	std::vector<float> dataVector(_memoryBlocks[0].blockSize.x * _memoryBlocks[0].blockSize.y);

	cs.getQueue().enqueueReadImage(
		sdr, CL_TRUE, _zeroOrigin, _memoryBlocks[0].clBlockRegion, 0, 0, dataVector.data());
	cs.getQueue().finish();

	return dataVector;
}

std::vector<float> Cortex::getChunkPredicts(ComputeSystem &cs)
{
	cl::Image2D sdr = utils::createImage2D(cs, _predictBlock.blockSize, CL_R, CL_FLOAT);

	_setValuesFromWinnersKernel.setArg(0, _predictBlock.chunkPredicts);
	_setValuesFromWinnersKernel.setArg(1, sdr);
	_setValuesFromWinnersKernel.setArg(2, _predictBlock.chunkSize);

	cl::NDRange range = cl::NDRange(_predictBlock.numChunks.x, _predictBlock.numChunks.y);
	cs.getQueue().enqueueNDRangeKernel(_setValuesFromWinnersKernel, cl::NullRange, range);
	cs.getQueue().finish();

	std::vector<float> dataVector(_predictBlock.blockSize.x * _predictBlock.blockSize.y);

	cs.getQueue().enqueueReadImage(
		sdr, CL_TRUE, _zeroOrigin, _predictBlock.clBlockRegion, 0, 0, dataVector.data());
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