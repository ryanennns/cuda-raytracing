#pragma once

#include "../include/Vector3D.h"
#include "cuda_runtime.h"
#include <cmath>
#include <stdio.h>

class Vector3D
{
public:
    double x, y, z;
    __host__ __device__ Vector3D()
    {
        this->x = 0.0;
        this->y = 0.0;
        this->z = 0.0;
    }

    __host__ __device__ Vector3D(Vector3D* v)
    {
        this->x = v->x;
        this->y = v->y;
        this->z = v->z;
    }

    __host__ __device__ Vector3D(double x, double y, double z)
    {
        this->x = x;
        this->y = y;
        this->z = z;
    }

    __host__ double magnitude()
    {
        return std::sqrt(x * x + y * y + z * z);
    }

    __host__ Vector3D normalize()
    {
        double magnitude = this->magnitude();

        return Vector3D(
            this->x / magnitude,
            this->y / magnitude,
            this->z / magnitude
        );
    }

    __host__ __device__ Vector3D add(const Vector3D& v)
    {
        return Vector3D(
            this->x + v.x,
            this->y + v.y,
            this->z + v.z
        );
    }

    __host__ __device__ Vector3D subtract(const Vector3D& v)
    {
        return Vector3D(
            this->x - v.x,
            this->y - v.y,
            this->z - v.z
        );
    }

    __host__ __device__ Vector3D multiply(double scalar)
    {
        return Vector3D(
            this->x * scalar,
            this->y * scalar,
            this->z * scalar
        );
    }

    __host__ __device__ double dotProduct(const Vector3D& v)
    {
        return (
            (this->x * v.x) +
            (this->y * v.y) +
            (this->z * v.z)
            );
    }

    __host__ __device__ Vector3D crossProduct(const Vector3D& v)
    {
        return Vector3D(
            (this->y * v.z) - (this->z * v.y),
            (this->z * v.x) - (this->x * v.z),
            (this->x * v.y) - (this->y * v.x)
        );
    }

    __host__ double distanceBetween(const Vector3D& v)
    {
        double x = v.x - this->x;
        double y = v.y - this->y;
        double z = v.z - this->z;

        return std::sqrt(x * x + y * y + z * z);
    }

    __host__ void consoleDisplay()
    {
        printf("(%lf, %lf, %lf)\n", this->x, this->y, this->z);
    }

    __host__ __device__ Vector3D negative()
    {
        return Vector3D(
            -this->x,
            -this->y,
            -this->z
        );
    }
};