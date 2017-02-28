// ===============
// predict_block.h
// ===============

#ifndef PREDICT_BLOCK_H
#define PREDICT_BLOCK_H

#include "utils/utils.h"

class PredictBlock
{
public:

	cl_int numColumns;
	cl_int numNodesInColumn;
	cl_int numNodesInField;

	cl_int  winnersSize;
	cl_int2 patternMemoriesSize;
	cl_int2 patternSumsSize;

	cl::size_t<3> clWinnersRegion;

	cl::Image1D winners;
	cl::Image2D patternMemories;
	cl::Image2D patternSums;

	void initialize(
		utils::Vec3i &blockDims,
		int numPatterns)
	{
		numColumns =
			static_cast<cl_int>(blockDims.x * blockDims.y);

		winnersSize = numColumns;

		patternMemoriesSize = {
			static_cast<cl_int>(numColumns),
			static_cast<cl_int>(numPatterns)};

		clWinnersRegion[0] = winnersSize;
		clWinnersRegion[1] = 1;
		clWinnersRegion[2] = 1;
	}
};

#endif
