#include "Vector3D.cu"
#include "Ray.cu"

const int WIDTH = 600;
const int HEIGHT = 300;

class ViewPort
{
public:
	Vector3D eye;

	ViewPort()
	{
		this->eye = Vector3D(0, 0, 0);
	}

	ViewPort(Vector3D eye, double width, double height)
	{
		this->eye = eye;
	}

    Ray** generateRays()
    {
        int width = WIDTH;
        int height = HEIGHT;
        double aspectRatio = static_cast<double>(width) / static_cast<double>(height);
        double zOffset = 1;

        Ray** rays = new Ray*[width];
        for (int x = 0; x < width; x++)
        {
            rays[x] = new Ray[height];
            for (int y = 0; y < height; y++)
            {
                double normalizedX = (x + 0.5) / width;
                double normalizedY = (y + 0.5) / height;

                normalizedX = (2 * normalizedX) - 1;
                normalizedY = (2 * normalizedY) - 1;

                normalizedX *= aspectRatio; 

                rays[x][y] = *(new Ray(this->eye, Vector3D(normalizedX, normalizedY, zOffset)));
            }
        }

        return rays;
    }
};