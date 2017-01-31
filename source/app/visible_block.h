// ===============
// visible_block.h
// ===============

#ifndef VISIBLE_BLOCK_H
#define VISIBLE_BLOCK_H

#include "utils/utils.h"

class VisibleBlock
{
public:
	void initialize(
		utils::Vec2ui32 &bSize,
		utils::Vec2ui32 &cSize,
		utils::Vec2ui32 &vSize,
		utils::Vec2ui32 &fSize,
		float &lRate)
	{
		blockSize = {
			static_cast<cl_int>(bSize.x),
			static_cast<cl_int>(bSize.y)};

		chunkSize = {
			static_cast<cl_int>(cSize.x),
			static_cast<cl_int>(cSize.y)};

		visibleSize = {
			static_cast<cl_int>(vSize.x),
			static_cast<cl_int>(vSize.y)};

		fieldSize = {
			static_cast<cl_int>(fSize.x),
			static_cast<cl_int>(fSize.y)};

		weightSize = {
			static_cast<cl_int>(blockSize.x),
			static_cast<cl_int>(blockSize.y),
			static_cast<cl_int>(fieldSize.x * fieldSize.y)};

		initWeightRange = {
			static_cast<cl_float>(0.00f),
			static_cast<cl_float>(1.00f)};

		numChunks = {
			blockSize.x / chunkSize.x,
			blockSize.y / chunkSize.y};

		fieldOffset = {
			visibleSize.x / numChunks.x,
			visibleSize.y / numChunks.y};

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

		clVisibleRegion[0] = visibleSize.x;
		clVisibleRegion[1] = visibleSize.y;
		clVisibleRegion[2] = 1;

		clWeightRegion[0] = weightSize.x;
		clWeightRegion[1] = weightSize.y;
		clWeightRegion[2] = weightSize.z;

		learningRate = lRate;
	}

	cl_int2 blockSize;
	cl_int2 chunkSize;
	cl_int2 visibleSize;
	cl_int2 fieldSize;
	cl_int3 weightSize;

	cl_int2 numChunks;
	cl_int2 fieldOffset;
	cl_int2 fieldStart;
	cl_int2 fieldStop;

	cl_float2 initWeightRange;

	cl::Image2D inputs;
	cl::Image2D sums;
	cl::Image2D outputs;
	cl::Image3D weights;

	cl::size_t<3> clBlockRegion;
	cl::size_t<3> clChunkRegion;
	cl::size_t<3> clVisibleRegion;
	cl::size_t<3> clWeightRegion;

	float learningRate;
};

#endif
