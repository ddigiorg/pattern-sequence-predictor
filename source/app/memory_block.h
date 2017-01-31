// ==============
// memory_block.h
// ==============

#ifndef MEMORY_BLOCK_H
#define MEMORY_BLOCK_H

#include "utils/utils.h"

class MemoryBlock
{
public:
	void initialize(
		utils::Vec2ui32 &bSize,
		utils::Vec2ui32 &cSize,
		float &lRate)
	{
		blockSize = {
			static_cast<cl_int>(bSize.x),
			static_cast<cl_int>(bSize.y)};

		chunkSize = {
			static_cast<cl_int>(cSize.x),
			static_cast<cl_int>(cSize.y)};

		weightSize = {
			static_cast<cl_int>(blockSize.x),
			static_cast<cl_int>(blockSize.y),
			static_cast<cl_int>(blockSize.x * blockSize.y)};

		initWeightRange = {
			static_cast<cl_float>(0.00f),
			static_cast<cl_float>(0.10f)};

		numChunks = {
			blockSize.x / chunkSize.x,
			blockSize.y / chunkSize.y};

		clBlockRegion[0] = blockSize.x;
		clBlockRegion[1] = blockSize.y;
		clBlockRegion[2] = 1;

		clChunkRegion[0] = numChunks.x;
		clChunkRegion[1] = numChunks.y;
		clChunkRegion[2] = 1;

		clWeightRegion[0] = weightSize.x;
		clWeightRegion[1] = weightSize.y;
		clWeightRegion[2] = weightSize.z;

		learningRate = lRate;
	}

	cl_int2 blockSize;
	cl_int2 chunkSize;
	cl_int3 weightSize;

	cl_int2 numChunks;

	cl_float2 initWeightRange;

	cl::Image2D chunkWinners;
	cl::Image2D sums;
	cl::Image3D weights;

	cl::size_t<3> clBlockRegion;
	cl::size_t<3> clChunkRegion;
	cl::size_t<3> clWeightRegion;

	float learningRate;
};

#endif
