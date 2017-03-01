// =========
// region.cl
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

kernel void initRandomMemories(
	write_only image3d_t memories,
	float2 initialRange,
	uint2 seed)
{
	uint2 valueSeed = seed + (uint2)(
		get_global_id(0) * 12 + 76 + get_global_id(2) * 3,
		get_global_id(1) * 21 + 42 + get_global_id(2) * 7) * 12;

	int3 indexMemory = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));

	float valueMemory = randFloat(&valueSeed) * (initialRange.y - initialRange.x) + initialRange.x;

	write_imagef(memories, (int4)(indexMemory, 0), (float4)(valueMemory, 0.0f, 0.0f, 0.0f));
}

kernel void initFieldCenters2D(
	write_only image2d_t fieldCenters,
	int2 sizeColumns,
	int2 sizeFields,
	int2 fieldOffset)
{
	int indexColumnFlat = get_global_id(0);

	int indexColumnX = indexColumnFlat % sizeColumns.x;
	int indexColumnY = indexColumnFlat / sizeColumns.x;

	int indexFieldCenterX = indexColumnX * fieldOffset.x + sizeFields.x / 2;
	int indexFieldCenterY = indexColumnY * fieldOffset.y + sizeFields.y / 2;

	write_imagef(fieldCenters, (int2)(indexColumnFlat, 0), (float4)(indexFieldCenterX, 0.0f, 0.0f, 0.0f));
	write_imagef(fieldCenters, (int2)(indexColumnFlat, 1), (float4)(indexFieldCenterY, 0.0f, 0.0f, 0.0f));
}

kernel void setInputSums(
	read_only image2d_t inputs,
	read_only image2d_t fieldCenters,
	read_only image3d_t inputMemories,
	write_only image2d_t inputSums,
	int2 sizeInputs,
	int2 sizeFields,
	int2 fieldStart,
	int2 fieldStop)
{
	int indexColumn = get_global_id(0);
	int indexNode = get_global_id(1);

	int indexFieldCenterX = read_imagef(fieldCenters, sampler, (int2)(indexColumn, 0)).x;
	int indexFieldCenterY = read_imagef(fieldCenters, sampler, (int2)(indexColumn, 1)).x;
    
	float sum = 0.0f;

	for (int y = fieldStart.y; y <= fieldStop.y; y++)
	{
		for (int x = fieldStart.x; x <= fieldStop.x; x++)
		{
			int2 indexInput = (int2)(indexFieldCenterX, indexFieldCenterY) + (int2)(x, y);

			if (inBounds(indexInput, (int2)(0, 0), sizeInputs))
			{
				int2 indexField = -fieldStart + (int2)(x, y);

				int3 indexMemory = (int3)(
					indexColumn,
					indexNode,
					indexField.x + sizeFields.x * indexField.y);

				float input = read_imagef(inputs, sampler, indexInput).x;

				float memory = read_imagef(inputMemories, sampler, (int4)(indexMemory, 0)).x;

				float delta = input - memory;

				sum += delta * delta;
			}
		}
	}

	write_imagef(inputSums, (int2)(indexColumn, indexNode), sum);
}

kernel void setPattern(
	read_only image2d_t inputSums,
	write_only image1d_t pattern,
	int numNodesInColumn)
{
	int indexColumn = get_global_id(0);

	float valueMin = 99999.0f;
	int indexNodeMin = 0;

	for (int n = 0; n < numNodesInColumn; n++)
	{
		float sum = read_imagef(inputSums, sampler, (int2)(indexColumn, n)).x;

		if (sum < valueMin)
		{
			valueMin = sum;
			indexNodeMin = n;
		}
	}

	write_imagef(pattern, indexColumn, indexNodeMin);
}

kernel void setPatternSums(
	read_only image1d_t pattern,
	read_only image2d_t patternMemories,
	write_only image1d_t patternSums,
	int numColumns)
{
	int indexPattern = get_global_id(0);

	int sum = 0;

	for (int c = 0; c < numColumns; c++)
	{
		int winner = read_imagef(pattern, sampler, c).x;

		int memory = read_imagef(patternMemories, sampler, (int2)(c, indexPattern)).x;

		if (winner == memory)
			sum++;
	}

	write_imagef(patternSums, indexPattern, sum);
}

kernel void setPatternIndex(
	read_only image1d_t patternSums,
	read_only image1d_t patternLearnIndex,
	write_only image1d_t patternIndex,
	write_only image1d_t patternLearnFlag,
	int numPatterns,
	int numColumns)
{
	write_imagef(patternLearnFlag, 0, 1);

	int learnIndex = read_imagef(patternLearnIndex, sampler, 0).x;

	write_imagef(patternIndex, 0, learnIndex);

	for (int p = 0; p < numPatterns; p++)
	{
		int sum = read_imagef(patternSums, sampler, p).x;

		if (sum == numColumns)
		{
			write_imagef(patternLearnFlag, 0, 0);
			write_imagef(patternIndex, 0, p);
		}
	}
}

kernel void setSequence(
	read_only image1d_t patternIndex,
	read_only image1d_t sequenceR,
	write_only image1d_t sequenceW,
	int sizeSequence)
{
	for (int s = sizeSequence - 2; s >= 0; s--)
	{
		int sequencePrev = read_imagef(sequenceR, sampler, s).x;

		write_imagef(sequenceW, s + 1, sequencePrev);
	}

	int patternIdx = read_imagef(patternIndex, sampler, 0).x;

	write_imagef(sequenceW, 0, patternIdx);
}

kernel void setSequenceSums(
	read_only image1d_t sequence,
	read_only image2d_t sequenceMemories,
	write_only image1d_t sequenceSums,
	int sizeSequence)
{
	int indexSequence = get_global_id(0);

	int sum = 0;

	for (int s = 0; s < sizeSequence - 1; s++)
	{
		int seq = read_imagef(sequence, sampler, s).x;

		int memory = read_imagef(sequenceMemories, sampler, (int2)(s, indexSequence)).x;

		if (seq == memory)
			sum++;
	}

	write_imagef(sequenceSums, indexSequence, sum);
}

kernel void setSequenceIndex(
	read_only image1d_t sequenceSums,
	read_only image1d_t sequenceLearnIndex,
	write_only image1d_t sequenceIndex,
	write_only image1d_t sequenceLearnFlag,
	int numSequences,
	int sizeSequence)
{
	write_imagef(sequenceLearnFlag, 0, 1);

	int learnIndex = read_imagef(sequenceLearnIndex, sampler, 0).x;

	write_imagef(sequenceIndex, 0, learnIndex);

	for (int s = 0; s < numSequences; s++)
	{
		int sum = read_imagef(sequenceSums, sampler, s).x;

		if (sum == sizeSequence - 1)
		{
			write_imagef(sequenceLearnFlag, 0, 0);
			write_imagef(sequenceIndex, 0, s);
		}
	}
}

kernel void setOutputs(
	read_only image1d_t predictMemories,
	read_only image1d_t sequenceIndex,
	read_only image2d_t patternMemories,
	read_only image2d_t fieldCenters,
	read_only image3d_t inputMemories,
	write_only image2d_t outputs,
	int2 sizeOutputs,
	int2 sizeFields,
	int2 fieldStart,
	int2 fieldStop)
{
	int indexColumn = get_global_id(0);

	int sequenceIdx = read_imagef(sequenceIndex, sampler, 0).x;
	int predictIdx = read_imagef(predictMemories, sampler, sequenceIdx).x;

	int indexWinner = read_imagef(patternMemories, sampler, (int2)(indexColumn, predictIdx)).x;

	int indexFieldCenterX = read_imagef(fieldCenters, sampler, (int2)(indexColumn, 0)).x;
	int indexFieldCenterY = read_imagef(fieldCenters, sampler, (int2)(indexColumn, 1)).x;

	for (int y = fieldStart.y; y <= fieldStop.y; y++)
	{
		for (int x = fieldStart.x; x <= fieldStop.x; x++)
		{
			int2 indexOutput = (int2)(indexFieldCenterX, indexFieldCenterY) + (int2)(x, y);

			if (inBounds(indexOutput, (int2)(0, 0), sizeOutputs))
			{
				int2 indexField = -fieldStart + (int2)(x, y);

				int3 indexInputMemories = (int3)(
					indexColumn,
					indexWinner,
					indexField.x + sizeFields.x * indexField.y);

				float inputMemory = read_imagef(inputMemories, sampler, (int4)(indexInputMemories, 0)).x;

				write_imagef(outputs, indexOutput, (float4)(inputMemory, 0.0f, 0.0f, 0.0f));
			}
		}
	}
}

kernel void learnInputMemories(
	read_only image1d_t pattern,
	read_only image2d_t inputs,
	read_only image2d_t fieldCenters,
	read_only image3d_t inputMemoriesPrev,
	write_only image3d_t inputMemories,
	int2 sizeOutputs,
	int2 sizeFields,
	int2 fieldStart,
	int2 fieldStop,
	float learningRate)
{
	int indexColumn = get_global_id(0);

	int indexWinner = read_imagef(pattern, sampler, indexColumn).x;

	int indexFieldCenterX = read_imagef(fieldCenters, sampler, (int2)(indexColumn, 0)).x;
	int indexFieldCenterY = read_imagef(fieldCenters, sampler, (int2)(indexColumn, 1)).x;

	for (int y = fieldStart.y; y <= fieldStop.y; y++)
	{
		for (int x = fieldStart.x; x <= fieldStop.x; x++)
		{
			int2 indexOutput = (int2)(indexFieldCenterX, indexFieldCenterY) + (int2)(x, y);

			if (inBounds(indexOutput, (int2)(0, 0), sizeOutputs))
			{
				int2 indexField = -fieldStart + (int2)(x, y);

				float input = read_imagef(inputs, sampler, indexOutput).x;

//				for (int i = -1; i <= 1; i++)
//				{
//					int n = winnerIndex + i;

//					if (n >= 0 && n < 10)
//					{
						int3 indexInputMemory = (int3)(
							indexColumn,
							indexWinner,
							indexField.x + sizeFields.x * indexField.y);

						float memoryPrev = read_imagef(inputMemoriesPrev, sampler, (int4)(indexInputMemory, 0)).x;

						float memory = memoryPrev + learningRate * (input - memoryPrev);

						write_imagef(inputMemories, (int4)(indexInputMemory, 0), (float4)(memory, 0.0f, 0.0f, 0.0f));
//					}
//			}
			}
		}
	}
}

kernel void learnPatternMemories(
	read_only image1d_t pattern,
	read_only image1d_t patternLearnFlag,
	read_only image1d_t patternLearnIndexR,
	write_only image1d_t patternLearnIndexW,
	write_only image2d_t patternMemories,
	int numColumns)
{
	int lFlag = read_imagef(patternLearnFlag, sampler, 0).x;

	if (lFlag == 1)
	{
		int learnIdx = read_imagef(patternLearnIndexR, sampler, 0).x;

		for (int c = 0; c < numColumns; c++)
		{
			int winner = read_imagef(pattern, sampler, c).x;

			write_imagef(patternMemories, (int2)(c, learnIdx), winner);
		}

		write_imagef(patternLearnIndexW, 0, learnIdx + 1);
	}
}

kernel void learnSequenceMemories(
	read_only image1d_t sequence,
	read_only image1d_t sequenceLearnFlag,
	read_only image1d_t sequenceLearnIndexR,
	write_only image1d_t sequenceLearnIndexW,
	write_only image2d_t sequenceMemories,
	write_only image1d_t predictMemories,
	int sizeSequence)
{
	int lFlag = read_imagef(sequenceLearnFlag, sampler, 0).x;

	if (lFlag == 1)
	{
		int learnIdx = read_imagef(sequenceLearnIndexR, sampler, 0).x;

		printf("%i\n", learnIdx);

		int pattern;

		for (int s = 1; s < sizeSequence; s++)
		{
			pattern = read_imagef(sequence, sampler, s).x;

			write_imagef(sequenceMemories, (int2)(s - 1, learnIdx), pattern);
		}

		pattern = read_imagef(sequence, sampler, 0).x;

		write_imagef(predictMemories, learnIdx, pattern);

		write_imagef(sequenceLearnIndexW, 0, learnIdx + 1);
	}
}
