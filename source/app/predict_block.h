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
		utils::Vec2ui32 &cSize,
		utils::Vec2ui32 &hSize,
		utils::Vec2ui32 &fSize)
	{
		blockSize = {
			static_cast<cl_int>(bSize.x),
			static_cast<cl_int>(bSize.y)};

		chunkSize = {
			static_cast<cl_int>(cSize.x),
			static_cast<cl_int>(cSize.y)};

		hiddenSize = {
			static_cast<cl_int>(hSize.x),
			static_cast<cl_int>(hSize.y)};

		fieldSize = {
			static_cast<cl_int>(fSize.x),
			static_cast<cl_int>(fSize.y)};

		numChunks = {
			blockSize.x / chunkSize.x,
			blockSize.y / chunkSize.y};

		fieldOffset = {
			hiddenSize.x / numChunks.x,
			hiddenSize.y / numChunks.y};

		fieldStart = {
			static_cast<cl_int>(-fieldSize.x / 2),
			static_cast<cl_int>(-fieldSize.y / 2)};

		if (fieldSize.x % 2 == 0)
			fieldStart.x++;

		if (fieldSize.y % 2 == 0)
			fieldStart.y++;

		fieldStop = {
			static_cast<cl_int>(fieldSize.x / 2),
			static_cast<cl_int>(fieldSize.y / 2)};

		clBlockRegion[0] = blockSize.x;
		clBlockRegion[1] = blockSize.y;
		clBlockRegion[2] = 1;

		clChunkRegion[0] = numChunks.x;
		clChunkRegion[1] = numChunks.y;
		clChunkRegion[2] = 1;

		clHiddenRegion[0] = hiddenSize.x;
		clHiddenRegion[1] = hiddenSize.y;
		clHiddenRegion[2] = 1;
	}

	cl_int2 blockSize;
	cl_int2 chunkSize;
	cl_int2 hiddenSize;
	cl_int2 fieldSize;

	cl_int2 numChunks;
	cl_int2 fieldOffset;
	cl_int2 fieldStart;
	cl_int2 fieldStop;

	cl_float2 initWeightRange;

	cl::Image2D votes;
	cl::Image2D predicts;

	cl::size_t<3> clBlockRegion;
	cl::size_t<3> clChunkRegion;
	cl::size_t<3> clHiddenRegion;

	float learningRate;
};

#endif
