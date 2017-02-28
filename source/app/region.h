// ========
// region.h
// ========

#ifndef REGION_H
#define REGION_H

#include "utils/utils.h"

#include "compute/compute-system.h"
#include "compute/compute-program.h"

#include <vector>
#include <random>
#include <iostream>

class Region
{
public:
	Region(std::mt19937 rng);

	void initialize(
		ComputeSystem &cs,
		ComputeProgram &cp,
		utils::Vec3i sizeNeurons,
		utils::Vec2i sizeInputs,
		utils::Vec2i sizeFields,
		int numHistories);

	void step(ComputeSystem& cs, std::vector<float> dataVector, bool learnFlag);

	void forecast(ComputeSystem& cs);

	std::vector<float> getInputs(ComputeSystem &cs);
	std::vector<float> getOutputs(ComputeSystem &cs);
//	std::vector<float> getWinners(ComputeSystem &cs);

private:
	void encode(ComputeSystem& cs);
	void predict(ComputeSystem& cs);
	void decode(ComputeSystem& cs);
	void learn(ComputeSystem& cs);

	std::mt19937 _rng;

	cl_float2 _initialMemoryRange = {0.0f, 0.0001f};

	cl_float4 _zeroFloat  = {0.0f, 0.0f, 0.0f, 0.0f};
	cl_float4 _halfFloat  = {0.5f, 0.0f, 0.0f, 0.0f};
	cl_uint4  _zeroUint32 = {0, 0, 0, 0};
//	cl_uint4  _maxUint32  = {4294967295, 4294967295, 4294967295, 4294967295};
//	cl_uint4  _maxUint32  = {10, 0, 0, 0};

	cl::size_t<3> _zeroOrigin;

	cl::NDRange _range;

//	int _count = 0;

	cl_int2 _sizeColumns;
	cl_int  _numColumns;
	cl_int  _numNodesInColumn;
	cl_int  _numNodesInField;
	cl_int  _numPatterns;
	cl_int  _numSequences;
	cl_int2 _sizeInputs;
	cl_int2 _sizeOutputs;
	cl_int2 _sizeFieldCenters;
	cl_int3 _sizeInputMemories;
	cl_int2 _sizeInputSums;
	cl_int  _sizePattern;
	cl_int2 _sizePatternMemories;
	cl_int  _sizePatternSums;
	cl_int  _sizeSequence;
	cl_int2 _sizeSequenceMemories;
	cl_int  _sizeSequenceSums;
	cl_int  _sizePredictMemories;
	cl_int2 _sizeFields;
	cl_int2 _fieldOffset;
	cl_int2 _fieldStart;
	cl_int2 _fieldStop;
	cl_int  _size1ValueImage;
	cl_float _inputLearningRate;

	cl::size_t<3> _clRegionInputs;
	cl::size_t<3> _clRegionOutputs;
	cl::size_t<3> _clRegionFieldCenters;
	cl::size_t<3> _clRegionInputMemories;
	cl::size_t<3> _clRegionInputSums;
	cl::size_t<3> _clRegionPattern;
	cl::size_t<3> _clRegionPatternMemories;
	cl::size_t<3> _clRegionPatternSums;
	cl::size_t<3> _clRegionSequence;
	cl::size_t<3> _clRegionSequenceMemories;
	cl::size_t<3> _clRegionSequenceSums;
	cl::size_t<3> _clRegionPredictMemories;
	cl::size_t<3> _clRegion1ValueImage;

	cl::Image2D _inputs;
	cl::Image2D _outputs;
	cl::Image2D _fieldCenters;
	cl::Image3D _inputMemories;
	cl::Image2D _inputSums;

	cl::Image1D _pattern;
	cl::Image2D _patternMemories;
	cl::Image1D _patternSums;

	cl::Image1D _patternIndex;
	cl::Image1D _patternLearnIndex;
	cl::Image1D _patternLearnFlag;

	cl::Image1D _sequence;
	cl::Image2D _sequenceMemories;
	cl::Image1D _sequenceSums;
	cl::Image1D _predictMemories;

	cl::Image1D _sequenceIndex;
	cl::Image1D _sequenceLearnIndex;
	cl::Image1D _sequenceLearnFlag;



	cl::Kernel _initRandomMemoriesKernel;
	cl::Kernel _initFieldCenters2DKernel;

	cl::Kernel _setInputSumsKernel;

	cl::Kernel _setPatternKernel;
	cl::Kernel _setPatternSumsKernel;
	cl::Kernel _setPatternIndexKernel;

	cl::Kernel _setSequenceKernel;
	cl::Kernel _setSequenceSumsKernel;
	cl::Kernel _setSequenceIndexKernel;

	cl::Kernel _setOutputsKernel;

	cl::Kernel _learnInputMemoriesKernel;
	cl::Kernel _learnPatternMemoriesKernel;
	cl::Kernel _learnSequenceMemoriesKernel;

};

#endif
