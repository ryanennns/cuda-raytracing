//#pragma once
//#include "cuda_runtime.h"
//
//class Vector3D {
//public:
//    double x, y, z;
//
//    Vector3D();
//    Vector3D(Vector3D* v);
//    __host__ __device__ Vector3D(double x, double y, double z);
//
//    double magnitude();
//    __device__ Vector3D normalize();
//    __device__ Vector3D add(const Vector3D& v);
//    __device__ Vector3D subtract(const Vector3D& v);
//    __device__ Vector3D multiply(double scalar);
//    __device__ double dotProduct(const Vector3D& v);
//    __device__ Vector3D crossProduct(const Vector3D& v);
//    double distanceBetween(const Vector3D& v);
//    void consoleDisplay();
//    __device__ Vector3D negative();
//};