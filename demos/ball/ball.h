// ======
// ball.h
// ======

#ifndef BALL_H
#define BALL_H

#include "utils/utils.h"

#include <vector>
#include <iostream>

class Ball
{
public:
	Ball(utils::Vec2i spaceSize)
	{
		_spaceSize = spaceSize;

		_position = utils::Vec2i(5, 5);

//		_position = utils::Vec2i(spaceSize.x / 2, spaceSize.y / 2);

		_initialPosition = _position;

		_velocity = utils::Vec2f(0.0f, 0.0f);

		_radius = 1;

		_pixelR.resize(_spaceSize.x * _spaceSize.y);
	}
 
	void reset()
	{
		_position.x = _initialPosition.x;
		_position.y = _initialPosition.y;

		_velocity.x = 0.0f;
		_velocity.y = 0.0f;
	}

	void step()
	{
		int i = 0;
		float value;
		float value2;
		for (int y = 0; y < _spaceSize.y; y++)
		{
			for (int x = 0; x < _spaceSize.x; x++)
			{
				i = x + (_spaceSize.x * y);

				// border
				value = (x == 0 || x == _spaceSize.x - 1 || y == 0 || y == _spaceSize.y - 1) ? 1.0f : 0.0f;

				// ball position
				if (
					x >= _position.x - _radius &&
					x <  _position.x + _radius &&
					y >= _position.y - _radius &&
					y <  _position.y + _radius)
					{
						value = 1.0f;
					}
					//else
					//  value = 0.0f;

					_pixelR[i] = value;
			}
		}

		/*
		_position.y += 3.0f;

		if (_position.y >= 45.0f)
			reset();
		*/

		if (_position.y <= 5.0f)
			_velocity.y = 3.0f * _acceleration;
		
		if (_position.y >= 42.0f)
			_velocity.y = 3.0f * -_acceleration;

		_position.y += _velocity.y;

		/*
		_velocity.y += _acceleration;

		_position.x += _velocity.x;
		_position.y += _velocity.y;

		bool onGround = _position.y + _radius >= _spaceSize.y - 1;

		if (onGround)
		{
			float vSquaredX = _velocity.x * _velocity.x;
			float vSquaredY = _velocity.y * _velocity.y;

			if (vSquaredX < 1.0 && vSquaredY < 2.0)
			{
				reset();
			}
			else
			{
				_velocity.y *= -0.75f;
				_position.y = _spaceSize.y - 1 - _radius;
			}
		}
		*/
		
	}

	std::vector<float> getPixelR()
	{
		return _pixelR;
	}

private:
	std::vector<float> _pixelR;

	utils::Vec2i _spaceSize;
	utils::Vec2i _position;
	utils::Vec2i _initialPosition;
	utils::Vec2f _velocity;
	float _acceleration = 1.0f;

	int _radius;
};

#endif
