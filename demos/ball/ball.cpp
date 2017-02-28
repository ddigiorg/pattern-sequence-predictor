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
	std::mt19937 rng(time(nullptr));

	// Setup SFML render window
	sf::RenderWindow window;
	utils::Vec2i displaySize(800, 600);

	window.create(sf::VideoMode(displaySize.x, displaySize.y), "FM - Ball", sf::Style::Default);

	// Setup OpenCL
	ComputeSystem cs;
	ComputeProgram cp;
	std::string kernels_cl = "source/app/region.cl"; // OpenCL kernel program

	cs.init(ComputeSystem::_gpu);
//	cs.printCLInfo();
	cp.loadProgramFromSourceFile(cs, kernels_cl);  // change to loadFromSourceFile

	utils::Vec3i sizeNeurons(16, 16, 10); // numColumns.x, numColumns.y, numNodesPerColumn
	utils::Vec2i sizeInputs(48, 48);      // numNodesInInput.x, numNodesInInput.y
	utils::Vec2i sizeFields(3, 3);        // numNodesInField.x, numNodesInField.y
	int numHistories = 1;

	Ball ball(sizeInputs);

	Region region(rng);

	region.initialize(cs, cp, sizeNeurons, sizeInputs, sizeFields, 1);

	Render2D renderInputs(sizeInputs);
	Render2D renderOutputs(sizeInputs);
//	Text2D textWinners(utils::Vec2i(blockSize.x, blockSize.y));

	renderInputs.setPosition(utils::Vec2i(375, 500));
	renderInputs.setScale(4.0f);

	renderOutputs.setPosition(utils::Vec2i(625, 500));
	renderOutputs.setScale(4.0f);

//	textWinners.setPosition(utils::Vec2i(375, 300));
//	textWinners.setScale(1.0f);

	bool quitFlag = false;
	bool stepFlag = false;

	while (!quitFlag)
	{
		sf::Event windowEvent;
		while (window.pollEvent(windowEvent))
		{
			if (windowEvent.type == sf::Event::Closed)
			{
				quitFlag = true;
				break;
			}

			if (sf::Keyboard::isKeyPressed(sf::Keyboard::Space))
			{
				stepFlag = true;
			}
		}
 
		if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
		{
			quitFlag = true;
			break;
		}

		if (stepFlag)
		{
			ball.step();

			region.step(cs, ball.getPixelR(), true);
//			region.learn(cs);

			renderInputs.setCheckered(region.getInputs(cs), sizeFields);
			renderOutputs.setCheckered(region.getOutputs(cs), sizeFields);
//			textWinners.setText(region.getWinners(cs));

			window.clear(sf::Color::Black);

			window.draw(renderInputs.getSprite());
			window.draw(renderOutputs.getSprite());
//			window.draw(textWinners.getText());

			stepFlag = false;
		}

		window.display();
	}
}
