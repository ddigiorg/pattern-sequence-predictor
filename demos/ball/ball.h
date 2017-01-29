// ======
// ball.h
// ======

#ifndef BALL_H
#define BALL_H

#include "utils/utils.h"

#include <vector>

class Ball
{
public:
	Ball(utils::Vec2ui32 spaceSize)
	{
		_spaceSize = spaceSize;

		_position = utils::Vec2ui32(spaceSize.x / 2, spaceSize.y / 2);

		_velocity = utils::Vec2f(0.0f, 0.0f);

		_radius = 2;

		_pixelR.resize(_spaceSize.x * _spaceSize.y);

		update();
	}
 
	void update()
	{
//		_velocity.x += 0.0f;
//		_velocity.y += _acceleration;

//		_position.x += _velocity.x;
//		_position.y += _velocity.y;

//		if (_position.y + _radius >= _spaceSize.y - 1)
//		{
//			_velocity.y = -0.99f;
//			_position.y = _spaceSize.y - 1 - _radius;
//		}

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
	}

	std::vector<float> getPixelR()
	{
		return _pixelR;
	}

private:
	std::vector<float> _pixelR;

	utils::Vec2ui32 _spaceSize;
	utils::Vec2ui32 _position;
	utils::Vec2f _velocity;
	float _acceleration = 0.5f;

	int _radius;
};

#endif
