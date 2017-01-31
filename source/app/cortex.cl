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

kernel void setSumsFromVisibles(
	read_only image2d_t visibles,
	read_only image3d_t weights,
	write_only image2d_t sums,
	int2 chunkSize,
	int2 visibleSize,
	int2 fieldSize,
	int2 fieldOffset)
{
	int2 blockPosition = (int2)(get_global_id(0), get_global_id(1));

	int2 chunkNumber = blockPosition / chunkSize;

	int2 fieldCenter = chunkNumber * fieldOffset + fieldSize / 2; 

	int2 fieldStart = -fieldSize / 2 - fieldSize % 2 + 1;

	int2 fieldStop = fieldSize / 2;

	float sum = 0.0f;

	for (int y = fieldStart.y; y <= fieldStop.y; y++)
	{
		for (int x = fieldStart.x; x <= fieldStop.x; x++)
		{
			int2 visiblePosition = fieldCenter + (int2)(x, y);

			if (inBounds(visiblePosition, (int2)(0, 0), visibleSize))
			{
				int2 fieldPosition = -fieldStart + (int2)(x, y);

				int3 weightPosition = (int3)(
					blockPosition.x,
					blockPosition.y,
					fieldPosition.x + fieldSize.x * fieldPosition.y);

				float visible = read_imagef(visibles, sampler, visiblePosition).x;

				float weight = read_imagef(weights, sampler, (int4)(weightPosition, 0)).x;

				float delta = visible - weight;

				sum += delta * delta;
			}
		}
	}

	write_imagef(sums, blockPosition, (float4)(sum, 0.0f, 0.0f, 0.0f));
}

kernel void setWinnersFromSums(
	read_only image2d_t sums,
	write_only image2d_t winners,
	int2 chunkSize)
{
	int2 chunkNumber = (int2)(get_global_id(0), get_global_id(1));

	float minValue = 99999.0f;
	int2 minIndex = (int2)(0, 0);

	for (int y = 0; y < chunkSize.y; y++)
	{
		for (int x = 0; x < chunkSize.x; x++)
		{
			int2 blockPosition = chunkNumber * chunkSize + (int2)(x, y);

			float sum = read_imagef(sums, sampler, blockPosition).x;

			if (sum < minValue)
			{
				minValue = sum;
				minIndex = (int2)(x, y);
			}
		}
	}

	write_imagef(winners, chunkNumber, (float4)((float)minIndex.x, (float)minIndex.y, 0.0f, 0.0f));
}

kernel void setSumsFromWinners(
	read_only image2d_t winners,
	read_only image3d_t weights,
	write_only image2d_t sums,
	int2 chunkSize)
{
	int2 blockPosition = (int2)(get_global_id(0), get_global_id(1));

	int2 blockSize = (int2)(get_global_size(0), get_global_size(1));

	float sum = 0.0f;

	for (int y = 0; y < blockSize.y; y++)
	{
		for (int x = 0; x < blockSize.x; x++)
		{
			int2 chunkNumber = (int2)(x, y) / chunkSize;

			int2 chunkPosition = (int2)(x, y) - (chunkNumber * chunkSize);

			int2 winner;
			winner.x = read_imagef(winners, sampler, chunkNumber).x;
			winner.y = read_imagef(winners, sampler, chunkNumber).y;

			int3 weightPosition = (int3)(
				blockPosition.x,
				blockPosition.y,
				x + blockSize.x * y);

			float value = (chunkPosition.x == winner.x && chunkPosition.y == winner.y) ? 1.0f : 0.0f;

			float weight = read_imagef(weights, sampler, (int4)(weightPosition, 0)).x;

			float delta = value - weight;

			sum += delta * delta;
		}
	}

	write_imagef(sums, blockPosition, (float4)(sum, 0.0f, 0.0f, 0.0f));
}

kernel void setVotesFromSums(
	read_only image2d_t sums,
	read_only image2d_t votesPrev,
	write_only image2d_t votes,
	int2 chunkSize)
{
	int2 chunkNumber = (int2)(get_global_id(0), get_global_id(1));

	float minValue = 99999.0f;
	int2 minIndex = (int2)(0, 0);

	for (int y = 0; y < chunkSize.y; y++)
	{
		for (int x = 0; x < chunkSize.x; x++)
		{
			int2 blockPosition = (int2)(x, y) + chunkNumber * chunkSize;

			float sum = read_imagef(sums, sampler, blockPosition).x;

			if (sum < minValue)
			{
				minValue = sum;
				minIndex = (int2)(x, y);
			}
		}
	}

	int2 votePosition = minIndex + chunkNumber * chunkSize;

	float votePrev = read_imagef(votesPrev, sampler, votePosition).x;

	float vote = votePrev + 1.0f;

	write_imagef(votes, votePosition, (float4)(vote, 0.0f, 0.0f, 0.0f));
}

kernel void setPredictsFromVotes(
	read_only image2d_t votes,
	write_only image2d_t predicts,
	int2 chunkSize)
{
	int2 chunkNumber = (int2)(get_global_id(0), get_global_id(1));

	float maxValue = -99999.0f;
	int2 maxIndex = (int2)(0, 0);

	for (int y = 0; y < chunkSize.y; y++)
	{
		for (int x = 0; x < chunkSize.x; x++)
		{
			int2 blockPosition = (int2)(x, y) + chunkNumber * chunkSize;

			float vote = read_imagef(votes, sampler, blockPosition).x;

			if (vote > maxValue)
			{
				maxValue = vote;
				maxIndex = (int2)(x, y);
			}
		}
	}

	write_imagef(predicts, chunkNumber, (float4)((float)maxIndex.x, (float)maxIndex.y, 0.0f, 0.0f));
}

kernel void setVisiblesFromPredicts(
	read_only image2d_t predicts,
	read_only image3d_t weights,
	write_only image2d_t visibles,
	int2 chunkSize,
	int2 visibleSize,
	int2 fieldSize,
	int2 fieldOffset)
{
	int2 chunkNumber = (int2)(get_global_id(0), get_global_id(1));

	int2 fieldCenter = chunkNumber * fieldOffset + fieldSize / 2; 

	int2 predict;
	predict.x = read_imagef(predicts, sampler, chunkNumber).x;
	predict.y = read_imagef(predicts, sampler, chunkNumber).y;

	int2 fieldStart = -fieldSize / 2 - fieldSize % 2 + 1;

	int2 fieldStop  = fieldSize / 2;

	for (int y = fieldStart.y; y <= fieldStop.y; y++)
	{
		for (int x = fieldStart.x; x <= fieldStop.x; x++)
		{
			int2 visiblePosition = fieldCenter + (int2)(x, y);

			if (inBounds(visiblePosition, (int2)(0, 0), visibleSize))
			{
				int2 fieldPosition = -fieldStart + (int2)(x, y);

				int3 weightPosition = (int3)(
					predict.x + chunkNumber.x * chunkSize.x,
					predict.y + chunkNumber.y * chunkSize.y,
					fieldPosition.x + fieldSize.x * fieldPosition.y);

				float weight = read_imagef(weights, sampler, (int4)(weightPosition, 0)).x;

				write_imagef(visibles, visiblePosition, (float4)(weight, 0.0f, 0.0f, 0.0f));
			}
		}
	}
}

kernel void learnWeightsFromPreviousVisibles(
	read_only image2d_t visibles,
	read_only image2d_t winners,
	read_only image3d_t weightsPrev,
	write_only image3d_t weights,
	int2 chunkSize,
	int2 visibleSize,
	int2 fieldSize,
	int2 fieldOffset,
	float learningRate)
{
	int2 chunkNumber = (int2)(get_global_id(0), get_global_id(1));

	int2 fieldCenter = chunkNumber * fieldOffset + fieldSize / 2; 

	int2 winner;
	winner.x = read_imagef(winners, sampler, chunkNumber).x;
	winner.y = read_imagef(winners, sampler, chunkNumber).y;

	int2 fieldStart = -fieldSize / 2 - fieldSize % 2 + 1;

	int2 fieldStop  = fieldSize / 2;

	for (int y = fieldStart.y; y <= fieldStop.y; y++)
	{
		for (int x = fieldStart.x; x <= fieldStop.x; x++)
		{
			int2 visiblePosition = fieldCenter + (int2)(x, y);

			if (inBounds(visiblePosition, (int2)(0, 0), visibleSize))
			{
				int2 fieldPosition = -fieldStart + (int2)(x, y);

				int3 weightPosition = (int3)(
					winner.x + chunkNumber.x * chunkSize.x,
					winner.y + chunkNumber.y * chunkSize.y,
					fieldPosition.x + fieldSize.x * fieldPosition.y);

				float visible = read_imagef(visibles, sampler, visiblePosition).x;

				float weightPrev = read_imagef(weightsPrev, sampler, (int4)(weightPosition, 0)).x;

				float weight = weightPrev + learningRate * (visible - weightPrev);

				write_imagef(weights, (int4)(weightPosition, 0), (float4)(weight, 0.0f, 0.0f, 0.0f));
			}
		}
	}
}

kernel void learnWeightsFromPreviousWinners(
	read_only image2d_t winnersPrev,
	read_only image2d_t winners,
	read_only image3d_t weightsPrev,
	write_only image3d_t weights,
	int2 chunkSize,
	float learningRate)
{
	int2 chunkNumber = (int2)(get_global_id(0), get_global_id(1));

	int2 blockSize = (int2)(get_global_size(0) * chunkSize.x, get_global_size(1) * chunkSize.y);

	int2 winner;
	winner.x = read_imagef(winners, sampler, chunkNumber).x;
	winner.y = read_imagef(winners, sampler, chunkNumber).y;

	for (int y = 0; y < blockSize.y; y++)
	{
		for (int x = 0; x < blockSize.x; x++)
		{
			int2 chunkNumberPrev = (int2)(x, y) / chunkSize;

			int2 chunkPositionPrev = (int2)(x, y) - (chunkNumberPrev * chunkSize);

			int2 winnerPrev;
			winnerPrev.x = read_imagef(winnersPrev, sampler, chunkNumberPrev).x;
			winnerPrev.y = read_imagef(winnersPrev, sampler, chunkNumberPrev).y;

			int3 weightPosition = (int3)(
				winner.x + chunkNumber.x * chunkSize.x,
				winner.y + chunkNumber.y * chunkSize.y,
				x + blockSize.x * y);

			float value = (chunkPositionPrev.x == winnerPrev.x && chunkPositionPrev.y == winnerPrev.y) ? 1.0f : 0.0f;

			float weightPrev = read_imagef(weightsPrev, sampler, (int4)(weightPosition, 0)).x;

			float weight = weightPrev + learningRate * (value - weightPrev);

//			if (chunkNumber.x == 7 && chunkNumber.y == 7 && chunkNumberPrev.x == 7 && chunkNumberPrev.y == 7)
//			{
//				printf("%f  ", value);
//				printf("%f  ", weightPrev);
//				printf("%f\n", weight);
//			}

			write_imagef(weights, (int4)(weightPosition, 0), (float4)(weight, 0.0f, 0.0f, 0.0f));
		}
	}

//	if (chunkNumber.x == 7 && chunkNumber.y == 7)
//	{
//		printf("\n");
//	}

}

kernel void setValuesFromWinners(
	read_only image2d_t winners,
	write_only image2d_t values,
	int2 chunkSize)
{
	int2 chunkNumber = (int2)(get_global_id(0), get_global_id(1));

	for (int y = 0; y < chunkSize.y; y++)
	{
		for (int x = 0; x < chunkSize.x; x++)
		{
			int2 valuePosition = (int2)(x, y) + chunkNumber * chunkSize;

			int2 winner;
			winner.x = read_imagef(winners, sampler, chunkNumber).x;
			winner.y = read_imagef(winners, sampler, chunkNumber).y;

			float value = (x == winner.x && y == winner.y) ? 1.0f : 0.0f;

			write_imagef(values, valuePosition, (float4)(value, 0.0f, 0.0f, 0.0f));
		}
	}
}
