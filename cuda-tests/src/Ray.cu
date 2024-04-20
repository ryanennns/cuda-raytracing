#pragma once

#include "Vector3D.cu"

class Ray
{
private:
public:
	Vector3D A;
	Vector3D B;
	__host__ __device__ Ray()
	{
		this->A = Vector3D(0, 0, 0);
		this->B = Vector3D(1, 0, 0);
	}

	__host__ __device__ Ray(Vector3D origin, Vector3D direction)
	{
		this->A = origin;
		this->B = direction;
	}

	__host__ __device__ Ray(Vector3D* origin, Vector3D* direction)
	{
		this->A = Vector3D(origin);
		this->B = Vector3D(direction);
	}

	__host__ __device__ Vector3D getA()
	{
		return this->A;
	}

	__host__ __device__ Vector3D getOrigin()
	{
		return this->A;
	}

	__host__ __device__ Vector3D getB()
	{
		return this->B;
	}

	__host__ __device__ Vector3D getDirection()
	{
		return this->B.subtract(this->A);
	}

	__host__ __device__ Vector3D evaluate(double t)
	{
		return this->A.add(this->getDirection().multiply(t));
	}

	__host__ __device__ Ray transform(Vector3D v)
	{
		return Ray(this->A.add(v), this->B.add(v));
	}
};