#pragma once

#include <cmath>

#include "Vector3D.cu"
#include "Ray.cu"
#include "Rgb.cu"

class Triangle
{
public:
	Vector3D A, B, C;
	Rgb colour;
	double specularCoefficient;

	__host__ __device__ Triangle(Vector3D A, Vector3D B, Vector3D C)
	{
		this->A = A;
		this->B = B;
		this->C = C;
		this->colour = Rgb(128, 0, 0);
		this->specularCoefficient = 2;
	}

	__host__ __device__ Triangle(Vector3D A, Vector3D B, Vector3D C, Rgb colour)
	{
		this->A = A;
		this->B = B;
		this->C = C;
		this->colour = colour;
		this->specularCoefficient = 2;
	}

	__host__ __device__ Triangle(Vector3D A, Vector3D B, Vector3D C, Rgb colour, double specularCoefficient)
	{
		this->A = A;
		this->B = B;
		this->C = C;
		this->colour = colour;
		this->specularCoefficient = specularCoefficient;
	}

	__host__ __device__ Vector3D getA()
	{
		return this->A;
	}

	__host__ __device__ Vector3D getB()
	{
		return this->B;
	}

	__host__ __device__ Vector3D getC()
	{
		return this->C;
	}

	__host__ __device__ Rgb getColour()
	{
		return this->colour;
	}

	__host__ __device__ void setA(Vector3D A)
	{
		this->A = A;
	}

	__host__ __device__ void setB(Vector3D B)
	{
		this->B = B;
	}

	__host__ __device__ void setC(Vector3D C)
	{
		this->C = C;
	}

	__host__ __device__ void setColour(Rgb colour)
	{
		this->colour = colour;
	}

	__host__ __device__ Vector3D normal()
	{
		Vector3D AB = B.subtract(A);
		Vector3D AC = C.subtract(A);
		Vector3D normal = AB.crossProduct(AC);
		normal.normalize();
		return normal;
	}

	__host__ __device__ Vector3D intersections(Ray ray)
	{
		//const double epsilon = 1e-12;
		Vector3D returnVector = Vector3D();

		Vector3D normal = this->normal();
		double NdotRayDirection = normal.dotProduct(ray.getDirection());

		if (NdotRayDirection == 0.0) {
			return returnVector;
		}

		double d = -normal.dotProduct(this->getA());
		double t = -((normal.dotProduct(ray.getOrigin())) + d)
			/ NdotRayDirection;

		Vector3D planeIntersection = ray.evaluate(t);
		if (
			this->isPointInTriangle(planeIntersection)
			&& t > 0
			&& this->verifyIntersection(planeIntersection, ray.getOrigin())
			) {
			return planeIntersection;
		}

		return returnVector;
	}

	__host__ __device__ bool isPointInTriangle(Vector3D point)
	{
		Vector3D edge0 = B.subtract(A);
		Vector3D edge1 = C.subtract(B);
		Vector3D edge2 = A.subtract(C);

		Vector3D C0 = point.subtract(A);
		Vector3D C1 = point.subtract(B);
		Vector3D C2 = point.subtract(C);

		if (this->normal().dotProduct(edge0.crossProduct(C0)) > 0 &&
			this->normal().dotProduct(edge1.crossProduct(C1)) > 0 &&
			this->normal().dotProduct(edge2.crossProduct(C2)) > 0) {

			return true;
		}

		return false;
	}

	__host__ __device__ void transform(Vector3D translation, Vector3D rotation)
	{
		this->A = this->A.add(translation);
		this->B = this->B.add(translation);
		this->C = this->C.add(translation);
	}

	__host__ __device__ void translate(Vector3D translation)
	{
		this->A = this->A.add(translation);
		this->B = this->B.add(translation);
		this->C = this->C.add(translation);
	}

	__host__ __device__ void rotate(Vector3D rotation)
	{
		// todo -- implement rotation
	}

	__host__ __device__ bool verifyIntersection(Vector3D a, Vector3D b)
	{
		// todo -- revisit this in case it causes problems down the line
		return fabs(a.x - b.x) > 1e-12
			|| fabs(a.y - b.y) > 1e-12
			|| fabs(a.z - b.z) > 1e-12;
	}

	__host__ __device__ Vector3D getNormal(Vector3D point)
	{
		return this->normal();
	}

	__host__ __device__ double getSpecularCoefficient()
	{
		return this->specularCoefficient;
	}
};