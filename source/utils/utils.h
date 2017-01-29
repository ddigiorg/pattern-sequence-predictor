// =======
// utils.h
// =======

#ifndef UTILS_H
#define UTILS_H

#include "compute/compute-system.h"

namespace utils
{

	typedef struct Vec2ui32
	{
		uint32_t x, y;

		Vec2ui32(){};

		Vec2ui32(uint32_t initX, uint32_t initY)
		{
			x = initX;
			y = initY;
		}
	} Vec2ui32;

	typedef struct Vec2f
	{
		float x, y;

		Vec2f(){};

		Vec2f(float initX, float initY)
		{
			x = initX;
			y = initY;
		}
	} Vec2f;

	typedef struct Vec4f
	{
		float r, b, g, a;

		Vec4f(){};

		Vec4f(float initR, float initB, float initG, float initA)
		{
			r = initR;
			g = initG;
			b = initB;
			a = initA;
		}
	} Vec4f;

	inline cl::Image2D createImage2D(ComputeSystem &cs, cl_int2 size, cl_channel_order channelOrder, cl_channel_type channelType)
	{	
		return cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(channelOrder, channelType), size.x, size.y);
	}

	inline cl::Image3D createImage3D(ComputeSystem &cs, cl_int3 size, cl_channel_order channelOrder, cl_channel_type channelType)
	{
		return cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(channelOrder, channelType), size.x, size.y, size.z);
	}
}

#endif
