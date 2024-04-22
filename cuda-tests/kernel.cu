#include <SDL.h>
#undef main
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <chrono>

#include "src/HitDetection.cu"
#include "src/Vector3D.cu"
#include "src/Triangle.cu"
#include "src/Rgb.cu"
#include "src/Ray.cu"
#include "src/ViewPort.cu"

void set_pixel(SDL_Surface* surface, int x, int y, Rgb& color)
{
    if (x >= 0 && x < surface->w && y >= 0 && y < surface->h) {
        Uint32* pixels = (Uint32*)surface->pixels;
        Uint32 pixelColor = SDL_MapRGB(
            surface->format,
            color.getRed(),
            color.getGreen(),
            color.getBlue()
        );

        pixels[(y * surface->w) + x] = pixelColor;
    }
}

void cleanup(SDL_Window* window)
{
    SDL_DestroyWindow(window);

    SDL_Quit();
}

__global__ void _processColumn(Ray* rays, Triangle* triangle, HitDetection* hitDetection)
{
    int idx = threadIdx.x;

    hitDetection[idx] = triangle->intersections(rays[idx]);
}

void renderRow(HitDetection* hitDetections, int index, SDL_Surface* screen)
{
    for (int i = 0; i < HEIGHT; i++) {
        if (hitDetections[i].hit)
            set_pixel(screen, index, i, Rgb(255, 0, 0));
        else
            set_pixel(screen, index, i, Rgb(255, 255, 255));
    }
}

HitDetection* processColumn(Ray** rays, Triangle triangle, int index)
{
    HitDetection* hitDetections = new HitDetection[HEIGHT];
    Ray* rayRow = rays[index];

    HitDetection* d_hitDetections = nullptr;
    Triangle* d_triangle = nullptr;
    Ray* d_rays = nullptr;

    cudaMalloc(&d_rays, HEIGHT * sizeof(Ray));
    cudaMalloc(&d_hitDetections, HEIGHT * sizeof(HitDetection));
    cudaMalloc(&d_triangle, sizeof(triangle));

    cudaMemcpy(
        d_rays,
        rayRow,
        HEIGHT * sizeof(Ray),
        cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        hitDetections,
        d_hitDetections,
        HEIGHT * sizeof(HitDetection),
        cudaMemcpyDeviceToHost
    );
    cudaMemcpy(
        d_triangle,
        &triangle,
        sizeof(triangle),
        cudaMemcpyHostToDevice
    );

    _processColumn <<<1, HEIGHT>>> (d_rays, d_triangle, d_hitDetections);

    cudaDeviceSynchronize();

    cudaMemcpy(hitDetections, d_hitDetections, HEIGHT * sizeof(HitDetection), cudaMemcpyDeviceToHost);
    
    cudaFree(d_rays);
    cudaFree(d_hitDetections);
    cudaFree(d_triangle);

    return hitDetections;
}

void freeHitDetectionColumn(HitDetection* hitDetections)
{
	delete[] hitDetections;
}

int main()
{
    SDL_Window* window = SDL_CreateWindow("SDL Tutorial", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, WIDTH, HEIGHT, SDL_WINDOW_SHOWN);
    SDL_Surface* screenSurface = SDL_GetWindowSurface(window);

    //=================================================================

    ViewPort v = ViewPort();
    Ray** rays = v.generateRays();
    Triangle triangle = Triangle(
        Vector3D(5, 0, 10),
        Vector3D(-5, 5, 10),
        Vector3D(-5, -5, 10)
    );

    while (true)
    {
        triangle.translate(Vector3D(0, 0.1, 0));
        auto start = std::chrono::high_resolution_clock::now();

        for (int x = 0; x < WIDTH; x++)
        {
            HitDetection* hd = processColumn(rays, triangle, x);
            renderRow(hd, x, screenSurface);
            freeHitDetectionColumn(hd);
            SDL_UpdateWindowSurface(window);
        }

        auto finish = std::chrono::high_resolution_clock::now();
        int ms = (int)std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count();
        printf("FPS: %lf\n", 1 / (ms * 0.001));
    }

    SDL_Event e;
    bool quit = false;
    while (quit == false)
    {
        while (SDL_PollEvent(&e))
        {
            if (e.type == SDL_QUIT) {
                quit = true;
            }
        }
    }

    return 0;
}