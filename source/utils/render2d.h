// ==========
// render2d.h
// ==========

#ifndef RENDER2D_H
#define RENDER2D_H

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <iostream>

class Render2D
{
public:
	Render2D(utils::Vec2i size)
	{
		_size = size;
		_image.create(_size.x, _size.y);
	}

	void setLuminance(std::vector<float> imageData)
	{
		for (int y = 0; y < _size.y; y++)
		{
			for (int x = 0; x < _size.x; x++)
			{
 				unsigned int i = x + _size.x * y;

				_color.r = _color.g = _color.b = 255.0f * imageData[i];

				_image.setPixel(x, y, _color);
			}
		}

		_texture.loadFromImage(_image);
		_sprite.setTexture(_texture);
		_sprite.setOrigin(sf::Vector2f(_texture.getSize().x * 0.5f, _texture.getSize().y * 0.5f));
	}

	void setCheckered(std::vector<float> imageData, utils::Vec2i checkerSize)
	{
		for (int y = 0; y < _size.y; y++)
		{
			for (int x = 0; x < _size.x; x++)
			{
 				unsigned int i = x + _size.x * y;

				int xx = (x - (x / (2 * checkerSize.x)) * (2 * checkerSize.x)) / checkerSize.x;
				int yy = (y - (y / (2 * checkerSize.y)) * (2 * checkerSize.y)) / checkerSize.y;

				if (xx ^ yy)
				{
					_color.r = 255.0f * imageData[i];
					_color.g = _color.b = 0.0f;
				}
				else
				{
					_color.b = 255.0f * imageData[i];
					_color.r = _color.g = 0.0f;
					_image.setPixel(x, y, _color);
				}

				_image.setPixel(x, y, _color);
			}
		}

		_texture.loadFromImage(_image);
		_sprite.setTexture(_texture);
		_sprite.setOrigin(sf::Vector2f(_texture.getSize().x * 0.5f, _texture.getSize().y * 0.5f));
	}


	void setPosition(utils::Vec2i position)
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
	utils::Vec2i _size;

	sf::Color _color;
	sf::Image _image;
	sf::Texture _texture;
	sf::Sprite _sprite;
};

#endif
