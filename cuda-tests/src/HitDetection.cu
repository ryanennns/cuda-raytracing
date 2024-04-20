#pragma once

#include "Vector3D.cu"

class HitDetection
{
public:
	Vector3D intersection;
	bool hit;

	__host__ __device__ HitDetection()
	{
		this->intersection = Vector3D();
		hit = false;
	}

	__host__ __device__ HitDetection(Vector3D intersection)
	{
		this->intersection = intersection;
		hit = true;
	}
};