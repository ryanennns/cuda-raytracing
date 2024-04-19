#pragma once

#include "cuda_runtime.h"

class Rgb
{
public:
	int r;
	int g;
	int b;

	__host__ __device__ Rgb()
	{
		r = 0;
		g = 0;
		b = 0;
	}

	__host__ __device__ Rgb(int r, int g, int b, bool placeholder)
	{
		this->r = r;
		this->g = g;
		this->b = b;
	}

	__host__ __device__ Rgb(int r, int g, int b)
	{
		this->r = r;
		this->g = g;
		this->b = b;
	}

	void setRed(int r)
	{
		this->r = r;
	}
	
	void setGreen(int g)
	{
		this->g = g;
	}

	void setBlue(int b)
	{
		this->b = b;
	}

	int getRed()
	{
		return r;
	}

	int getGreen()
	{
		return g;
	}

	int getBlue()
	{
		return b;
	}

	__device__ void _setRed(int r)
	{
		this->r = r;
	}
	__device__ void _setGreen(int g)
	{
		this->g = g;
	}
	__device__ void _setBlue(int b)
	{
		this->b = b;
	}
	__device__ int _getRed()
	{
		return r;
	}
	__device__ int _getGreen()
	{
		return g;
	}
	__device__ int _getBlue()
	{
		return b;
	}

	__device__ Rgb operator*(Rgb r)
	{
		return Rgb(this->r * r.r, this->g * r.g, this->b * r.b);
	}

	__device__ Rgb operator*(int r)
	{
		return Rgb(this->r * r, this->g * r, this->b * r);
	}
};