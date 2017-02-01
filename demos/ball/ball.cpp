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
	utils::Vec2ui32 displaySize(1000, 600);

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

	Ball ball(visibleSize);

	Cortex cortex(rng);

	cortex.addVisibleBlock(
		utils::Vec2ui32(128, 128), // blockSize
		utils::Vec2ui32(  8,   8), // chunkSize
		utils::Vec2ui32( 48,  48), // visibleSize
		utils::Vec2ui32(  3,   3), // fieldSize
		0.25f);                    // learningRate

	cortex.addMemoryBlocks(
		1,                         // numBlocks
		utils::Vec2ui32(128, 128), // blockSize
		utils::Vec2ui32(  8,   8), // chunkSize
		0.25f);                    // learningRate

	cortex.addPredictBlock(
		utils::Vec2ui32(128, 128),  // blockSize
		utils::Vec2ui32(  8,   8)); // chunkSize

	cortex.initialize(cs, cp);

	std::vector<float> ballData = ball.getPixelR();

	Render2D renderWinnersOldest(blockSize);
	Render2D renderVisibleInputs(visibleSize);
	Render2D renderWinners0(blockSize);
	Render2D renderPredicts(blockSize);
	Render2D renderVisibleOutputs(visibleSize);

	renderWinnersOldest.setPosition(utils::Vec2ui32(150, 250));
	renderWinnersOldest.setScale(2.0f);

	renderVisibleInputs.setPosition(utils::Vec2ui32(450, 500));
	renderVisibleInputs.setScale(3.0f);

	renderWinners0.setPosition(utils::Vec2ui32(450, 250));
	renderWinners0.setScale(2.0f);

	renderPredicts.setPosition(utils::Vec2ui32(750, 250));
	renderPredicts.setScale(2.0f);

	renderVisibleOutputs.setPosition(utils::Vec2ui32(750, 500));
	renderVisibleOutputs.setScale(3.0f);

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

			if (sf::Keyboard::isKeyPressed(sf::Keyboard::Space))
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
			ball.step();

			ballData = ball.getPixelR();

			cortex.setVisibleInputs(cs, ballData);

			cortex.step(cs, true);

			renderWinnersOldest.setPixelsR(cortex.getChunkWinnersOldest(cs));
			renderVisibleInputs.setPixelsR(cortex.getVisibleInputs(cs));
//			renderWinners0.setPixelsR(cortex.getChunkWinners(cs, 0));
//			renderPredicts.setPixelsR(cortex.getChunkPredicts(cs));
			renderWinners0.setPixelsRB(cortex.getChunkWinners(cs, 0), cortex.getChunkPredicts(cs));
			renderVisibleOutputs.setPixelsR(cortex.getVisibleOutputs(cs));

			window.draw(renderWinnersOldest.getSprite());
			window.draw(renderVisibleInputs.getSprite());
			window.draw(renderWinners0.getSprite());
//			window.draw(renderPredicts.getSprite());
			window.draw(renderVisibleOutputs.getSprite());

			cont = false;
		}

		//glDisplay.drawFPS();
		window.display();
	}
}
