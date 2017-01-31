// ===============
// predict_block.h
// ===============

#ifndef PREDICT_BLOCK_H
#define PREDICT_BLOCK_H

#include "utils/utils.h"

class PredictBlock
{
public:
	void initialize(
		utils::Vec2ui32 &bSize,
		utils::Vec2ui32 &cSize)
	{
		blockSize = {
			static_cast<cl_int>(bSize.x),
			static_cast<cl_int>(bSize.y)};

		chunkSize = {
			static_cast<cl_int>(cSize.x),
			static_cast<cl_int>(cSize.y)};

		numChunks = {
			blockSize.x / chunkSize.x,
			blockSize.y / chunkSize.y};

		clBlockRegion[0] = blockSize.x;
		clBlockRegion[1] = blockSize.y;
		clBlockRegion[2] = 1;

		clChunkRegion[0] = numChunks.x;
		clChunkRegion[1] = numChunks.y;
		clChunkRegion[2] = 1;
	}

	cl_int2 blockSize;
	cl_int2 chunkSize;

	cl_int2 numChunks;

	cl_float2 initWeightRange;

	cl::Image2D votes;
	cl::Image2D chunkPredicts;

	cl::size_t<3> clBlockRegion;
	cl::size_t<3> clChunkRegion;
};

#endif
