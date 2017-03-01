// ========
// ball.cpp
// ========

#include "ball.h"

#include "utils/render2d.h"
#include "utils/text2d.h"

#include "compute/compute-system.h"
#include "compute/compute-program.h"

#include "app/region.h"

#include <iostream>
#include <vector>
#include <random>
#include <time.h>

int main()
{
	std::mt19937 rng(time(nullptr));  // for initializing random weights
	srand(time(NULL));  // for getting random floats in utils

	// Setup SFML render window
	sf::RenderWindow window;
	utils::Vec2i displaySize(800, 600);

	window.create(sf::VideoMode(displaySize.x, displaySize.y), "DM - Ball", sf::Style::Default);

	// Setup OpenCL
	ComputeSystem cs;
	ComputeProgram cp;
	std::string kernels_cl = "source/app/region.cl";

	cs.init(ComputeSystem::_gpu);
//	cs.printCLInfo();
	cp.loadProgramFromSourceFile(cs, kernels_cl);  // change to loadFromSourceFile

	utils::Vec3i sizeNeurons(48, 48, 2); // numColumns.x, numColumns.y, numNodesPerColumn
	utils::Vec2i sizeInputs(48, 48);     // numNodesInInput.x, numNodesInInput.y
	utils::Vec2i sizeFields(1, 1);       // numNodesInField.x, numNodesInField.y
	int numHistories = 1;

	Ball ball(sizeInputs);

	Region region(rng);

	region.initialize(cs, cp, sizeNeurons, sizeInputs, sizeFields, 1);

	Render2D scene(sizeInputs);

	scene.setPosition(utils::Vec2i(400, 300));
	scene.setScale(6.0f);

	bool quit = false;
	bool pause = false;

	while (!quit)
	{
		// Handle SFML window events
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
				pause = false;
			}

			if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
			{
				quit = true;
				break;
			}
		}
 
		if (!pause)
		{
			ball.step();

			region.encode(cs, ball.getPixelData());
			region.predict(cs);
			region.decode(cs);
			region.learn(cs);

//			for (int i = 0; i < 5; i++)
//			{
//				region.predict(cs);
//			}

			scene.setPixelData('g', false, region.getInputs(cs));
			scene.setPixelData('b', false, region.getOutputs(cs));

			window.clear(sf::Color::Black);

			window.draw(scene.getSprite());

			pause = true;
		}

		window.display();
	}
}
