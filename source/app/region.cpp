// ==========
// region.cpp
// ==========

#include "region.h"

#include <iostream>
#include <cstdlib>

Region::Region(std::mt19937 rng)
{
	_rng = rng;

	_zeroOrigin[0] = 0;
	_zeroOrigin[1] = 0;
	_zeroOrigin[2] = 0;
}

void Region::initialize(
	ComputeSystem &cs,
	ComputeProgram &cp,
	utils::Vec3i sizeNeurons,
	utils::Vec2i sizeInputs,
	utils::Vec2i sizeFields,
	int numHistories)
{
	_sizeColumns = {
		static_cast<cl_int>(sizeNeurons.x),
		static_cast<cl_int>(sizeNeurons.y)};

	_numColumns =
		static_cast<cl_int>(sizeNeurons.x * sizeNeurons.y);

	_numNodesInColumn =
		static_cast<cl_int>(sizeNeurons.z);

	_numNodesInField =
		static_cast<cl_int>(sizeFields.x * sizeFields.y);

	_numPatterns =
		static_cast<cl_int>(10000);

	_numSequences =
		static_cast<cl_int>(10000);

	_sizeInputs = {
		static_cast<cl_int>(sizeInputs.x),
		static_cast<cl_int>(sizeInputs.y)};

	_sizeOutputs = {
		static_cast<cl_int>(sizeInputs.x),
		static_cast<cl_int>(sizeInputs.y)};

	_sizeFieldCenters = {
		static_cast<cl_int>(_numColumns),
		static_cast<cl_int>(2)};
 
	_sizeInputMemories = {
		static_cast<cl_int>(_numColumns),
		static_cast<cl_int>(_numNodesInColumn),
		static_cast<cl_int>(_numNodesInField)};

	_sizeInputSums = {
		static_cast<cl_int>(_numColumns),
		static_cast<cl_int>(_numNodesInColumn)};

	_sizePattern = 
		static_cast<cl_int>(_numColumns);

	_sizePatternMemories = {
		static_cast<cl_int>(_numColumns),
		static_cast<cl_int>(_numPatterns)};

	_sizePatternSums = 
		static_cast<cl_int>(_numPatterns);

	_sizeSequence =
		static_cast<cl_int>(numHistories + 2);

	_sizeSequenceMemories = {
		static_cast<cl_int>(numHistories + 1),
		static_cast<cl_int>(_numSequences)};

	_sizeSequenceSums = 
		static_cast<cl_int>(_numSequences);

	_sizePredictMemories = 
		static_cast<cl_int>(_numSequences);

	_sizeFields = {
		static_cast<cl_int>(sizeFields.x),
		static_cast<cl_int>(sizeFields.y)};

	_fieldOffset = {
		static_cast<cl_int>(sizeInputs.x / sizeNeurons.x),
		static_cast<cl_int>(sizeInputs.y / sizeNeurons.y)};

	_fieldStart = {
		static_cast<cl_int>(-sizeFields.x / 2 - sizeFields.x % 2 + 1),
		static_cast<cl_int>(-sizeFields.y / 2 - sizeFields.y % 2 + 1)};
 
	_fieldStop = {
		static_cast<cl_int>(sizeFields.x / 2),
		static_cast<cl_int>(sizeFields.y / 2)};

	_size1ValueImage =
		static_cast<cl_int>(1);

	_inputLearningRate =
		static_cast<cl_float>(0.25);

	_clRegionInputs[0] = _sizeInputs.x;
 	_clRegionInputs[1] = _sizeInputs.y;
 	_clRegionInputs[2] = 1;

	_clRegionOutputs[0] = _sizeOutputs.x;
 	_clRegionOutputs[1] = _sizeOutputs.y;
 	_clRegionOutputs[2] = 1;

	_clRegionFieldCenters[0] = _sizeFieldCenters.x;
	_clRegionFieldCenters[1] = _sizeFieldCenters.y;
	_clRegionFieldCenters[2] = 1;

	_clRegionInputMemories[0] = _sizeInputMemories.x;
	_clRegionInputMemories[1] = _sizeInputMemories.y;
	_clRegionInputMemories[2] = _sizeInputMemories.z;

	_clRegionInputSums[0] = _sizeInputSums.x;
	_clRegionInputSums[1] = _sizeInputSums.y;
	_clRegionInputSums[2] = 1;

	_clRegionPattern[0] = _sizePattern;
	_clRegionPattern[1] = 1;
	_clRegionPattern[2] = 1;

	_clRegionPatternMemories[0] = _sizePatternMemories.x;
	_clRegionPatternMemories[1] = _sizePatternMemories.y;
	_clRegionPatternMemories[2] = 1;

	_clRegionPatternSums[0] = _sizePatternSums;
	_clRegionPatternSums[1] = 1;
	_clRegionPatternSums[2] = 1;

	_clRegionSequence[0] = _sizeSequence;
	_clRegionSequence[1] = 1;
	_clRegionSequence[2] = 1;

	_clRegionSequenceMemories[0] = _sizeSequenceMemories.x;
	_clRegionSequenceMemories[1] = _sizeSequenceMemories.y;
	_clRegionSequenceMemories[2] = 1;

	_clRegionSequenceSums[0] = _sizeSequenceSums;
	_clRegionSequenceSums[1] = 1;
	_clRegionSequenceSums[2] = 1;

	_clRegionPredictMemories[0] = _sizePredictMemories;
	_clRegionPredictMemories[1] = 1;
	_clRegionPredictMemories[2] = 1;

	_clRegion1ValueImage[0] = 1;
	_clRegion1ValueImage[1] = 1;
	_clRegion1ValueImage[2] = 1;

	// Define Images
	_inputs             = utils::createImage2D(cs, _sizeInputs,           CL_R, CL_FLOAT);
	_outputs            = utils::createImage2D(cs, _sizeOutputs,          CL_R, CL_FLOAT);
	_fieldCenters       = utils::createImage2D(cs, _sizeFieldCenters,     CL_R, CL_FLOAT);
	_inputMemories      = utils::createImage3D(cs, _sizeInputMemories,    CL_R, CL_FLOAT);
	_inputSums          = utils::createImage2D(cs, _sizeInputSums,        CL_R, CL_FLOAT);

	_pattern            = utils::createImage1D(cs, _sizePattern,          CL_R, CL_UNSIGNED_INT32);
	_patternMemories    = utils::createImage2D(cs, _sizePatternMemories,  CL_R, CL_UNSIGNED_INT32);
	_patternSums        = utils::createImage1D(cs, _sizePatternSums,      CL_R, CL_UNSIGNED_INT32);

	_patternIndex       = utils::createImage1D(cs, _size1ValueImage,      CL_R, CL_UNSIGNED_INT32);
	_patternLearnIndex  = utils::createImage1D(cs, _size1ValueImage,      CL_R, CL_UNSIGNED_INT32);
	_patternLearnFlag   = utils::createImage1D(cs, _size1ValueImage,      CL_R, CL_UNSIGNED_INT32);

	_sequence           = utils::createImage1D(cs, _sizeSequence,         CL_R, CL_UNSIGNED_INT32);
	_sequenceMemories   = utils::createImage2D(cs, _sizeSequenceMemories, CL_R, CL_UNSIGNED_INT32);
	_sequenceSums       = utils::createImage1D(cs, _sizeSequenceSums,     CL_R, CL_UNSIGNED_INT32);
	_predictMemories    = utils::createImage1D(cs, _sizePredictMemories,  CL_R, CL_UNSIGNED_INT32);

	_sequenceIndex      = utils::createImage1D(cs, _size1ValueImage,      CL_R, CL_UNSIGNED_INT32);
	_sequenceLearnIndex = utils::createImage1D(cs, _size1ValueImage,      CL_R, CL_UNSIGNED_INT32);
	_sequenceLearnFlag  = utils::createImage1D(cs, _size1ValueImage,      CL_R, CL_UNSIGNED_INT32);

	// Define Kernels
	_initRandomMemoriesKernel    = cl::Kernel(cp.getProgram(), "initRandomMemories");
	_initFieldCenters2DKernel    = cl::Kernel(cp.getProgram(), "initFieldCenters2D");

	_setInputSumsKernel          = cl::Kernel(cp.getProgram(), "setInputSums");

	_setPatternKernel            = cl::Kernel(cp.getProgram(), "setPattern");
	_setPatternSumsKernel        = cl::Kernel(cp.getProgram(), "setPatternSums");
	_setPatternIndexKernel       = cl::Kernel(cp.getProgram(), "setPatternIndex");

	_setSequenceKernel           = cl::Kernel(cp.getProgram(), "setSequence");
	_setSequenceSumsKernel       = cl::Kernel(cp.getProgram(), "setSequenceSums");
	_setSequenceIndexKernel      = cl::Kernel(cp.getProgram(), "setSequenceIndex");

	_setOutputsKernel            = cl::Kernel(cp.getProgram(), "setOutputs");

	_learnInputMemoriesKernel    = cl::Kernel(cp.getProgram(), "learnInputMemories");
	_learnPatternMemoriesKernel  = cl::Kernel(cp.getProgram(), "learnPatternMemories");
	_learnSequenceMemoriesKernel = cl::Kernel(cp.getProgram(), "learnSequenceMemories");


	// Initialize images
	cs.getQueue().enqueueFillImage(_inputs,             _zeroFloat,  _zeroOrigin, _clRegionInputs);
	cs.getQueue().enqueueFillImage(_outputs,            _zeroFloat,  _zeroOrigin, _clRegionOutputs);
	cs.getQueue().enqueueFillImage(_fieldCenters,       _zeroFloat,  _zeroOrigin, _clRegionFieldCenters);
	cs.getQueue().enqueueFillImage(_inputSums,          _zeroFloat,  _zeroOrigin, _clRegionInputSums);

	cs.getQueue().enqueueFillImage(_pattern,            _zeroUint32, _zeroOrigin, _clRegionPattern);
	cs.getQueue().enqueueFillImage(_patternMemories,    _zeroUint32, _zeroOrigin, _clRegionPatternMemories);
	cs.getQueue().enqueueFillImage(_patternSums,        _zeroUint32, _zeroOrigin, _clRegionPatternSums);

	cs.getQueue().enqueueFillImage(_patternIndex,       _zeroUint32, _zeroOrigin, _clRegion1ValueImage);
	cs.getQueue().enqueueFillImage(_patternLearnIndex,  _zeroUint32, _zeroOrigin, _clRegion1ValueImage);
	cs.getQueue().enqueueFillImage(_patternLearnFlag,   _zeroUint32, _zeroOrigin, _clRegion1ValueImage);

	cs.getQueue().enqueueFillImage(_sequence,           _zeroUint32, _zeroOrigin, _clRegionSequence);
	cs.getQueue().enqueueFillImage(_sequenceMemories,   _zeroUint32, _zeroOrigin, _clRegionSequenceMemories);
	cs.getQueue().enqueueFillImage(_sequenceSums,       _zeroUint32, _zeroOrigin, _clRegionSequenceSums);
	cs.getQueue().enqueueFillImage(_predictMemories,    _zeroUint32, _zeroOrigin, _clRegionPredictMemories);

	cs.getQueue().enqueueFillImage(_sequenceIndex,      _zeroUint32, _zeroOrigin, _clRegion1ValueImage);
	cs.getQueue().enqueueFillImage(_sequenceLearnIndex, _zeroUint32, _zeroOrigin, _clRegion1ValueImage);
	cs.getQueue().enqueueFillImage(_sequenceLearnFlag,  _zeroUint32, _zeroOrigin, _clRegion1ValueImage);

	// Initialize visible memories to random values
	std::uniform_int_distribution<int> seedDist(0, 999);

	cl_uint2 seed = {(cl_uint)seedDist(_rng), (cl_uint)seedDist(_rng)};

	_initRandomMemoriesKernel.setArg(0, _inputMemories);
	_initRandomMemoriesKernel.setArg(1, _initialMemoryRange);
	_initRandomMemoriesKernel.setArg(2, seed);

	_range = cl::NDRange(_sizeInputMemories.x, _sizeInputMemories.y, _sizeInputMemories.z);
	cs.getQueue().enqueueNDRangeKernel(_initRandomMemoriesKernel, cl::NullRange, _range);
	cs.getQueue().finish();

	// Initialize Visible Field Centers
	_initFieldCenters2DKernel.setArg(0, _fieldCenters);
	_initFieldCenters2DKernel.setArg(1, _sizeColumns);
	_initFieldCenters2DKernel.setArg(2, _sizeFields);
	_initFieldCenters2DKernel.setArg(3, _fieldOffset);

	_range = cl::NDRange(_numColumns);
	cs.getQueue().enqueueNDRangeKernel(_initFieldCenters2DKernel, cl::NullRange, _range);
	cs.getQueue().finish();
}

void Region::step(ComputeSystem& cs, std::vector<float> dataVector, bool learnFlag)
{
	cs.getQueue().enqueueWriteImage(_inputs, CL_TRUE, _zeroOrigin, _clRegionInputs, 0, 0, dataVector.data());
	cs.getQueue().finish();

	encode(cs);

	predict(cs);

	decode(cs);

	if (learnFlag)
		learn(cs);
}

void Region::forecast(ComputeSystem &cs)
{

}

std::vector<float> Region::getInputs(ComputeSystem &cs)
{
	std::vector<float> dataVector(_sizeInputs.x * _sizeInputs.y);

	cs.getQueue().enqueueReadImage(_inputs, CL_TRUE, _zeroOrigin, _clRegionInputs, 0, 0, dataVector.data());
	cs.getQueue().finish();

	return dataVector;
}

std::vector<float> Region::getOutputs(ComputeSystem &cs)
{
	std::vector<float> dataVector(_sizeOutputs.x * _sizeOutputs.y);

	cs.getQueue().enqueueReadImage(_outputs, CL_TRUE, _zeroOrigin, _clRegionOutputs, 0, 0, dataVector.data());
	cs.getQueue().finish();

	return dataVector;
}

/*
std::vector<float> Region::getWinners(ComputeSystem &cs)
{
	std::vector<float> dataVector(_numColumns);

	cs.getQueue().enqueueReadImage(_winnerNeurons, CL_TRUE, _zeroOrigin, _clRegionWinnerNeurons, 0, 0, dataVector.data());
	cs.getQueue().finish();

	return dataVector;
}
*/

void Region::encode(ComputeSystem &cs)
{
	_setInputSumsKernel.setArg(0, _inputs);
	_setInputSumsKernel.setArg(1, _fieldCenters);
	_setInputSumsKernel.setArg(2, _inputMemories);
	_setInputSumsKernel.setArg(3, _inputSums);
	_setInputSumsKernel.setArg(4, _sizeInputs);
	_setInputSumsKernel.setArg(5, _sizeFields);
	_setInputSumsKernel.setArg(6, _fieldStart);
	_setInputSumsKernel.setArg(7, _fieldStop);

	_range = cl::NDRange(_numColumns, _numNodesInColumn);
	cs.getQueue().enqueueNDRangeKernel(_setInputSumsKernel, cl::NullRange, _range);
	cs.getQueue().finish();

	_setPatternKernel.setArg(0, _inputSums);
	_setPatternKernel.setArg(1, _pattern);
	_setPatternKernel.setArg(2, _numNodesInColumn);

	_range = cl::NDRange(_numColumns);
	cs.getQueue().enqueueNDRangeKernel(_setPatternKernel, cl::NullRange, _range);
	cs.getQueue().finish();

	_setPatternSumsKernel.setArg(0, _pattern);
	_setPatternSumsKernel.setArg(1, _patternMemories);
	_setPatternSumsKernel.setArg(2, _patternSums);
	_setPatternSumsKernel.setArg(3, _numColumns);

	_range = cl::NDRange(_numPatterns);
	cs.getQueue().enqueueNDRangeKernel(_setPatternSumsKernel, cl::NullRange, _range);
	cs.getQueue().finish();

	_setPatternIndexKernel.setArg(0, _patternSums);
	_setPatternIndexKernel.setArg(1, _patternLearnIndex);
	_setPatternIndexKernel.setArg(2, _patternIndex);
	_setPatternIndexKernel.setArg(3, _patternLearnFlag);
	_setPatternIndexKernel.setArg(4, _numPatterns);
	_setPatternIndexKernel.setArg(5, _numColumns);

	_range = cl::NDRange(1);
	cs.getQueue().enqueueNDRangeKernel(_setPatternIndexKernel, cl::NullRange, _range);
	cs.getQueue().finish();
}

void Region::predict(ComputeSystem &cs)
{
	_setSequenceKernel.setArg(0, _patternIndex);
	_setSequenceKernel.setArg(1, _sequence);
	_setSequenceKernel.setArg(2, _sequence);
	_setSequenceKernel.setArg(3, _sizeSequence);

	_range = cl::NDRange(1);
	cs.getQueue().enqueueNDRangeKernel(_setSequenceKernel, cl::NullRange, _range);
	cs.getQueue().finish();

	_setSequenceSumsKernel.setArg(0, _sequence);
	_setSequenceSumsKernel.setArg(1, _sequenceMemories);
	_setSequenceSumsKernel.setArg(2, _sequenceSums);
	_setSequenceSumsKernel.setArg(3, _sizeSequence);

	_range = cl::NDRange(_numSequences);
	cs.getQueue().enqueueNDRangeKernel(_setSequenceSumsKernel, cl::NullRange, _range);
	cs.getQueue().finish();

	_setSequenceIndexKernel.setArg(0, _sequenceSums);
	_setSequenceIndexKernel.setArg(1, _sequenceLearnIndex);
	_setSequenceIndexKernel.setArg(2, _sequenceIndex);
	_setSequenceIndexKernel.setArg(3, _sequenceLearnFlag);
	_setSequenceIndexKernel.setArg(4, _numSequences);
	_setSequenceIndexKernel.setArg(5, _sizeSequence);

	_range = cl::NDRange(1);
	cs.getQueue().enqueueNDRangeKernel(_setSequenceIndexKernel, cl::NullRange, _range);
	cs.getQueue().finish();
}

void Region::decode(ComputeSystem &cs)
{
	_setOutputsKernel.setArg(0, _predictMemories);
	_setOutputsKernel.setArg(1, _sequenceIndex);
	_setOutputsKernel.setArg(2, _patternMemories);
	_setOutputsKernel.setArg(3, _fieldCenters);
	_setOutputsKernel.setArg(4, _inputMemories);
	_setOutputsKernel.setArg(5, _outputs);
	_setOutputsKernel.setArg(6, _sizeOutputs);
	_setOutputsKernel.setArg(7, _sizeFields);
	_setOutputsKernel.setArg(8, _fieldStart);
	_setOutputsKernel.setArg(9, _fieldStop);

	_range = cl::NDRange(_numColumns);
	cs.getQueue().enqueueNDRangeKernel(_setOutputsKernel, cl::NullRange, _range);
	cs.getQueue().finish();
}

void Region::learn(ComputeSystem &cs)
{
	_learnInputMemoriesKernel.setArg(0, _pattern);
	_learnInputMemoriesKernel.setArg(1, _inputs);
	_learnInputMemoriesKernel.setArg(2, _fieldCenters);
	_learnInputMemoriesKernel.setArg(3, _inputMemories);
	_learnInputMemoriesKernel.setArg(4, _inputMemories);
	_learnInputMemoriesKernel.setArg(5, _sizeOutputs);
	_learnInputMemoriesKernel.setArg(6, _sizeFields);
	_learnInputMemoriesKernel.setArg(7, _fieldStart);
	_learnInputMemoriesKernel.setArg(8, _fieldStop);
	_learnInputMemoriesKernel.setArg(9, _inputLearningRate);

	_range = cl::NDRange(_numColumns);
	cs.getQueue().enqueueNDRangeKernel(_learnInputMemoriesKernel, cl::NullRange, _range);
	cs.getQueue().finish();

	_learnPatternMemoriesKernel.setArg(0, _pattern);
	_learnPatternMemoriesKernel.setArg(1, _patternLearnFlag);
	_learnPatternMemoriesKernel.setArg(2, _patternLearnIndex);
	_learnPatternMemoriesKernel.setArg(3, _patternLearnIndex);
	_learnPatternMemoriesKernel.setArg(4, _patternMemories);
	_learnPatternMemoriesKernel.setArg(5, _numColumns);

	_range = cl::NDRange(1);
	cs.getQueue().enqueueNDRangeKernel(_learnPatternMemoriesKernel, cl::NullRange, _range);
	cs.getQueue().finish();

	_learnSequenceMemoriesKernel.setArg(0, _sequence);
	_learnSequenceMemoriesKernel.setArg(1, _sequenceLearnFlag);
	_learnSequenceMemoriesKernel.setArg(2, _sequenceLearnIndex);
	_learnSequenceMemoriesKernel.setArg(3, _sequenceLearnIndex);
	_learnSequenceMemoriesKernel.setArg(4, _sequenceMemories);
	_learnSequenceMemoriesKernel.setArg(5, _predictMemories);
	_learnSequenceMemoriesKernel.setArg(6, _sizeSequence);

	_range = cl::NDRange(1);
	cs.getQueue().enqueueNDRangeKernel(_learnSequenceMemoriesKernel, cl::NullRange, _range);
	cs.getQueue().finish();
}
