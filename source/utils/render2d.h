// ==========
// render2d.h
// ==========

#ifndef RENDER2D_H
#define RENDER2D_H

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

class Render2D
{
public:
	void setPixelsR(std::vector<float> imageData, utils::Vec2ui32 imageSize)
	{
		_image.create(imageSize.x, imageSize.y);

		for (int y = 0; y < imageSize.y; y++)
		{
			for (int x = 0; x < imageSize.x; x++)
			{
				sf::Color color;

 				unsigned int i = x + imageSize.x * y;

				color.r = 255.0f * imageData[i];

				_image.setPixel(x, y, color);
			}
		}

		_texture.loadFromImage(_image);
		_sprite.setTexture(_texture);
		_sprite.setOrigin(sf::Vector2f(_texture.getSize().x * 0.5f, _texture.getSize().y * 0.5f));
	}

	void setPosition(utils::Vec2ui32 position)
	{   
		_sprite.setPosition(position.x, position.y);
	}

	void setScale(float scale)
	{   
		_sprite.setScale(sf::Vector2f(scale, scale));
	}

	sf::Sprite getSprite()
	{   
		return _sprite;
	}

private:
	sf::Image _image;
	sf::Texture _texture;
	sf::Sprite _sprite;
};

#endif
