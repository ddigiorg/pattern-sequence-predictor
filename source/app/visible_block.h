// ===============
// visible_block.h
// ===============

#ifndef VISIBLE_BLOCK_H
#define VISIBLE_BLOCK_H

#include "utils/utils.h"

class VisibleBlock
{
public:
	cl_float learningRate;

	cl_int numColumns;
	cl_int numNodesInColumn;
	cl_int numNodesInField;

	cl_int3 blockSize;
	cl_int2 fieldSize;
	cl_int2 fieldOffset;
	cl_int2 fieldStart;
	cl_int2 fieldStop;

	cl_int2 visiblesSize;
	cl_int2 fieldCentersSize;
	cl_int2 contrastsSize;
	cl_int3 memoriesSize;

	cl::size_t<3> clVisiblesRegion;
	cl::size_t<3> clFieldCentersRegion;
	cl::size_t<3> clContrastsRegion;
	cl::size_t<3> clMemoriesRegion;

	cl::Image2D inputs;
	cl::Image2D outputs;
	cl::Image2D fieldCenters;
	cl::Image2D contrasts;
	cl::Image3D memories;

	void initialize(
		utils::Vec3i &blockDims,
		utils::Vec2i &visibleDims,
		utils::Vec2i &fieldDims,
		float &lRate)
	{
		learningRate =
			static_cast<cl_float>(lRate);

		numColumns =
			static_cast<cl_int>(blockDims.x * blockDims.y);

		numNodesInColumn =
			static_cast<cl_int>(blockDims.z);

		numNodesInField =
			static_cast<cl_int>(fieldDims.x * fieldDims.y);

		blockSize = {
			static_cast<cl_int>(blockDims.x),
			static_cast<cl_int>(blockDims.y),
			static_cast<cl_int>(blockDims.z)};

		fieldSize = {
			static_cast<cl_int>(fieldDims.x),
			static_cast<cl_int>(fieldDims.y)};

		fieldOffset = {
			static_cast<cl_int>(visibleDims.x / blockDims.x),
			static_cast<cl_int>(visibleDims.y / blockDims.y)};

		fieldStart = {
			static_cast<cl_int>(-fieldDims.x / 2 - fieldDims.x % 2 + 1),
			static_cast<cl_int>(-fieldDims.y / 2 - fieldDims.y % 2 + 1)};

		fieldStop = {
			static_cast<cl_int>(fieldDims.x / 2),
			static_cast<cl_int>(fieldDims.y / 2)};

		visiblesSize = {
			static_cast<cl_int>(visibleDims.x),
			static_cast<cl_int>(visibleDims.y)};

		fieldCentersSize = {
			static_cast<cl_int>(numColumns),
			static_cast<cl_int>(2)};

		contrastsSize = {
			static_cast<cl_int>(numColumns),
			static_cast<cl_int>(numNodesInColumn)};

		memoriesSize = {
			static_cast<cl_int>(numColumns),
			static_cast<cl_int>(numNodesInColumn),
			static_cast<cl_int>(numNodesInField)};

		clVisiblesRegion[0] = visiblesSize.x;
		clVisiblesRegion[1] = visiblesSize.y;
		clVisiblesRegion[2] = 1;

		clFieldCentersRegion[0] = fieldCentersSize.x;
		clFieldCentersRegion[2] = fieldCentersSize.y;
		clFieldCentersRegion[2] = 1;

		clContrastsRegion[0] = contrastsSize.x;
		clContrastsRegion[1] = contrastsSize.y;
		clContrastsRegion[2] = 1;

		clMemoriesRegion[0] = memoriesSize.x;
		clMemoriesRegion[1] = memoriesSize.y;
		clMemoriesRegion[2] = memoriesSize.z;
	}
};

#endif
