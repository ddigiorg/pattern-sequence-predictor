// ========
// ball.cpp
// ========

#include "ball.h"

#include "utils/render2d.h"

#include "compute/compute-system.h"
#include "compute/compute-program.h"

#include "app/cortex.h"

#include <iostream>
#include <vector>
#include <random>
#include <time.h>

int main()
{
	std::mt19937 rng(time(nullptr));

	// Initialize Compute-Render Engine
	utils::Vec2ui32 displaySize(800, 600);

	std::string feynman_cl = "source/app/cortex.cl"; // OpenCL kernel program

	sf::RenderWindow window;

	window.create(sf::VideoMode(displaySize.x, displaySize.y), "FM - Ball", sf::Style::Default);

	ComputeSystem cs;
	ComputeProgram cp;

	cs.init(ComputeSystem::_gpu);
//	cs.printCLInfo();

	cp.loadProgramFromSourceFile(cs, feynman_cl);  // change to loadFromSourceFile

	utils::Vec2ui32 visibleSize(48, 48);
	utils::Vec2ui32 blockSize(128, 128);

	Ball ballScene(visibleSize);

	Cortex cortex(rng);

	cortex.addVisibleBlock(
		utils::Vec2ui32(128, 128), // blockSize
		utils::Vec2ui32(  8,   8), // chunkSize
		utils::Vec2ui32( 48,  48), // visibleSize
		utils::Vec2ui32(  3,   3), // fieldSize
		0.05f);                    // learningRate

	cortex.addMemoryBlock(
		utils::Vec2ui32(128, 128), // blockSize
		utils::Vec2ui32(  8,   8), // chunkSize
		utils::Vec2ui32(128, 128), // hiddenSize
		utils::Vec2ui32(  8,   8), // fieldSize
		0.05f);                    // learningRate

	cortex.addPredictBlock(
		utils::Vec2ui32(128, 128),  // blockSize
		utils::Vec2ui32(  8,   8),  // chunkSize
		utils::Vec2ui32(128, 128),  // hiddenSize
		utils::Vec2ui32(  8,   8)); // fieldSize

	cortex.initialize(cs, cp);

	std::vector<float> ballSceneData = ballScene.getPixelR();

	Render2D renderVisibleInputs;
	Render2D renderChunkWinners0;
//	Render2D renderChunkPredicts;
	Render2D renderVisibleOutputs;

	renderVisibleInputs.setPosition(utils::Vec2ui32(100, 500));
	renderVisibleInputs.setScale(4.0f);

	renderChunkWinners0.setPosition(utils::Vec2ui32(100, 300));
	renderChunkWinners0.setScale(1.0f);

//	renderChunkPredicts.setPosition(utils::Vec2ui32(400, 300));
//	renderChunkPredicts.setScale(4.0f);

	renderVisibleOutputs.setPosition(utils::Vec2ui32(400, 500));
	renderVisibleOutputs.setScale(4.0f);

	bool quit = false;
	bool cont = false;
	while (!quit)
	{
		sf::Event windowEvent;
		while (window.pollEvent(windowEvent))
		{
			if (windowEvent.type == sf::Event::Closed)
			{
				quit = true;
				break;
			}

			if (sf::Keyboard::isKeyPressed(sf::Keyboard::P))
			{
				cont = true;
			}
		}
 
		if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
		{
			quit = true;
			break;
		}

		if (cont)
		{
			cortex.setVisibleInputs(cs, ballSceneData);

			cortex.step(cs, true);

			renderVisibleInputs.setPixelsR(cortex.getVisibleInputs(cs), visibleSize);
			renderChunkWinners0.setPixelsR(cortex.getChunkSDR(cs, 0), blockSize);
			renderVisibleOutputs.setPixelsR(cortex.getVisibleOutputs(cs), visibleSize);

			window.draw(renderVisibleInputs.getSprite());
			window.draw(renderChunkWinners0.getSprite());
			window.draw(renderVisibleOutputs.getSprite());

			cont = false;
		}

		//glDisplay.drawFPS();
		window.display();
	}
}
