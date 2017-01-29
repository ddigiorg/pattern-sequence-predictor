// =========
// cortex.cl
// =========

const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

bool inBounds(int2 position, int2 lowerBound, int2 upperBound)
{
	return	position.x >= lowerBound.x && position.x < upperBound.x && position.y >= lowerBound.y && position.y < upperBound.y;
}

// http://stackoverflow.com/questions/16064591/simple-opencl-random-generator
float randFloat(uint2* state)
{
	const float invMaxInt = 1.0f / 4294967296.0f;
	uint x = (*state).x * 17 + (*state).y * 13123;
	(*state).x = (x << 13) ^ x;
	(*state).y ^= (x << 7);

	uint tmp = x * (x * x * 15731 + 74323) + 871483;

	return convert_float(tmp) * invMaxInt;
}

kernel void initWeights(
	write_only image3d_t weights,
	float2 initWeightRange,
	uint2 seed)
{
	uint2 seedValue = seed + (uint2)(
		get_global_id(0) * 12 + 76 + get_global_id(2) * 3,
		get_global_id(1) * 21 + 42 + get_global_id(2) * 7) * 12;

	int3 weightPosition = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));

	float value = randFloat(&seedValue) * (initWeightRange.y - initWeightRange.x) + initWeightRange.x;

	write_imagef(weights, (int4)(weightPosition, 0), (float4)(value, 0.0f, 0.0f, 0.0f));
}

kernel void setSumsFromValues(
	read_only image2d_t values,
	read_only image3d_t weights,
	write_only image2d_t sums,
	int2 chunkSize,
	int2 valueSize,
	int2 fieldSize,
	int2 fieldStart,
	int2 fieldStop,
	int2 fieldOffset)
{
	int2 sumPosition = (int2)(get_global_id(0), get_global_id(1));

	int2 chunkPosition = sumPosition / chunkSize;

	int2 chunkCenter = chunkPosition * chunkSize + chunkSize / 2;

	int2 fieldCenter = chunkPosition * fieldOffset + fieldOffset / 2; 

	float sum = 0.0f;

	for (int dy = fieldStart.y; dy <= fieldStop.y; dy++)
	{
		for (int dx = fieldStart.x; dx <= fieldStop.x; dx++)
		{
			int2 valuePosition = fieldCenter + (int2)(dx, dy);

			if (inBounds(valuePosition, (int2)(0, 0), valueSize))
			{
				int3 weightPosition = (int3)(
					sumPosition.x,
					sumPosition.y,
					(dx - fieldStart.x) + fieldSize.x * (dy - fieldStart.y));

				float value = read_imagef(values, sampler, valuePosition).x;

				float weight = read_imagef(weights, sampler, (int4)(weightPosition, 0)).x;

				float delta = value - weight;

				sum += delta * delta;
			}
		}
	}

	write_imagef(sums, sumPosition, (float4)(sum, 0.0f, 0.0f, 0.0f));
}

kernel void setWinnersFromSums(
	read_only image2d_t sums,
	write_only image2d_t chunkWinners,
	int2 chunkSize)
{
	int2 chunkPosition = (int2)(get_global_id(0), get_global_id(1));

	int2 sumStartPosition = chunkPosition * chunkSize;

	float minValue = 99999.0f;
	int2 minIndex = (int2)(0, 0);

	for (int dy = 0; dy < chunkSize.y; dy++)
	{
		for (int dx = 0; dx < chunkSize.x; dx++)
		{
			int2 sumPosition = sumStartPosition + (int2)(dx, dy);

			float sum = read_imagef(sums, sampler, sumPosition).x;

			if (sum < minValue)
			{
				minValue = sum;
				minIndex = (int2)(dx, dy);
			}
		}
	}

	write_imagef(chunkWinners, chunkPosition, (float4)((float)minIndex.x, (float)minIndex.y, 0.0f, 0.0f));
}

kernel void setValuesFromPredicts(
	read_only image2d_t chunkPredicts,
	read_only image3d_t weights,
	write_only image2d_t values,
	int2 chunkSize,
	int2 valueSize,
	int2 fieldSize,
	int2 fieldStart,
	int2 fieldStop,
	int2 fieldOffset)
{
	int2 chunkPosition = (int2)(get_global_id(0), get_global_id(1));

	int2 fieldCenter = chunkPosition * fieldOffset + fieldOffset / 2; 

	int2 chunkPredict;
	chunkPredict.x = read_imagef(chunkPredicts, sampler, chunkPosition).x;
	chunkPredict.y = read_imagef(chunkPredicts, sampler, chunkPosition).y;

	for (int dy = fieldStart.y; dy <= fieldStop.y; dy++)
	{
		for (int dx = fieldStart.x; dx <= fieldStop.x; dx++)
		{
			int2 valuePosition = fieldCenter + (int2)(dx, dy);

			if (inBounds(valuePosition, (int2)(0, 0), valueSize))
			{
				int3 weightPosition = (int3)(
					chunkPredict.x + chunkPosition.x * chunkSize.x,
					chunkPredict.y + chunkPosition.y * chunkSize.y,
					(dx - fieldStart.x) + fieldSize.x * (dy - fieldStart.y));

				float weight = read_imagef(weights, sampler, (int4)(weightPosition, 0)).x;

				write_imagef(values, valuePosition, (float4)(weight, 0.0f, 0.0f, 0.0f));
			}
		}
	}
}

kernel void learnWeightsFromPreviousValues(
	read_only image2d_t values,
	read_only image2d_t chunkWinners,
	read_only image3d_t weightsPrev,
	write_only image3d_t weights,
	int2 chunkSize,
	int2 valueSize,
	int2 fieldSize,
	int2 fieldStart,
	int2 fieldStop,
	int2 fieldOffset,
	float learningRate)
{
	int2 chunkPosition = (int2)(get_global_id(0), get_global_id(1));

	int2 chunkCenter = chunkPosition * chunkSize + chunkSize / 2;

	int2 fieldCenter = chunkPosition * fieldOffset + fieldOffset / 2; 

	int2 chunkWinner;
	chunkWinner.x = read_imagef(chunkWinners, sampler, chunkPosition).x;
	chunkWinner.y = read_imagef(chunkWinners, sampler, chunkPosition).y;

	for (int dy = fieldStart.y; dy <= fieldStop.y; dy++)
	{
		for (int dx = fieldStart.x; dx <= fieldStop.x; dx++)
		{
			int2 valuePosition = fieldCenter + (int2)(dx, dy);

			if (inBounds(valuePosition, (int2)(0, 0), valueSize))
			{
				int3 weightPosition = (int3)(
					chunkWinner.x + chunkPosition.x * chunkSize.x,
					chunkWinner.y + chunkPosition.y * chunkSize.y,
					(dx - fieldStart.x) + fieldSize.x * (dy - fieldStart.y));

				float value = read_imagef(values, sampler, valuePosition).x;

				float weightPrev = read_imagef(weightsPrev, sampler, (int4)(weightPosition, 0)).x;

				float weight = weightPrev + learningRate * (value - weightPrev);

				write_imagef(weights, (int4)(weightPosition, 0), (float4)(weight, 0.0f, 0.0f, 0.0f));
			}
		}
	}
}

kernel void setValuesFromWinners(
	read_only image2d_t chunkWinners,
	write_only image2d_t values,
	int2 chunkSize)
{
	int2 chunkPosition = (int2)(get_global_id(0), get_global_id(1));

	int2 valueStartPosition = chunkPosition * chunkSize;

	for (int dy = 0; dy < chunkSize.y; dy++)
	{
		for (int dx = 0; dx < chunkSize.x; dx++)
		{
			int2 valuePosition = valueStartPosition + (int2)(dx, dy);

			int2 chunkWinner;
			chunkWinner.x = read_imagef(chunkWinners, sampler, chunkPosition).x;
			chunkWinner.y = read_imagef(chunkWinners, sampler, chunkPosition).y;

			float value = (dx == chunkWinner.x && dy == chunkWinner.y) ? 1.0f : 0.0f;

			write_imagef(values, valuePosition, (float4)(value, 0.0f, 0.0f, 0.0f));
		}
	}
}

