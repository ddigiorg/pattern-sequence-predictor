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
	Render2D(utils::Vec2ui32 imageSize)
	{
		_imageSize = imageSize;
		_image.create(_imageSize.x, _imageSize.y);
	}

	void setPixelsR(std::vector<float> imageData)
	{
		for (int y = 0; y < _imageSize.y; y++)
		{
			for (int x = 0; x < _imageSize.x; x++)
			{
 				unsigned int i = x + _imageSize.x * y;

				_color.r = 255.0f * imageData[i];

				_image.setPixel(x, y, _color);
			}
		}

		_texture.loadFromImage(_image);
		_sprite.setTexture(_texture);
		_sprite.setOrigin(sf::Vector2f(_texture.getSize().x * 0.5f, _texture.getSize().y * 0.5f));
	}

	void setPixelsRB(std::vector<float> imageDataR, std::vector<float> imageDataB)
	{
		for (int y = 0; y < _imageSize.y; y++)
		{
			for (int x = 0; x < _imageSize.x; x++)
			{
 				unsigned int i = x + _imageSize.x * y;

				_color.r = 255.0f * imageDataR[i];
				_color.b = 255.0f * imageDataB[i];

				_image.setPixel(x, y, _color);
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
	utils::Vec2ui32 _imageSize;

	sf::Color _color;
	sf::Image _image;
	sf::Texture _texture;
	sf::Sprite _sprite;
};

#endif
