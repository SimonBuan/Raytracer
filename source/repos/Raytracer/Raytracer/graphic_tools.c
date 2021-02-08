#include "graphic_tools.h"

#include <stdbool.h>

#include <SDL.h>
#include <SDL_image.h>

#include "obj_reader.h"

//Window used for rendering
SDL_Window* S_Window = NULL;

//Window renderer
SDL_Renderer* S_Renderer = NULL;

//Surface used to save displayed image to file
SDL_Surface* sshot = NULL;

//RGB and alpha masks used for screenshots
int rmask = 0x00FF0000;
int gmask = 0x0000FF00;
int bmask = 0x000000FF;
int amask = 0xFF000000;

//Graphics window dimensions
const int SCREEN_WIDTH = 100;
const int SCREEN_HEIGHT = 100;

//Initializes SDL, and the graphics window and renderer
//Sets render draw color to black
//Returns 1 on success, 0 on failure
int init_graphics(int w, int h)
{
	SDL_Log("Initalizing graphics\n");
	int success = true;

	//Initialize SDL
	if (SDL_Init(SDL_INIT_VIDEO) < 0)
	{
		SDL_Log("SDL could not initialize! SDL_Error: %s\n", SDL_GetError());
		success = false;
	}
	else
	{
		//Create window
		S_Window = SDL_CreateWindow("Raytracer", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, w, h, SDL_WINDOW_SHOWN);
		if (S_Window == NULL)
		{
			SDL_Log("Window could not be created! SDL_Error: %s\n", SDL_GetError());
			success = false;
		}
		else
		{
			//Create renderer for window
			S_Renderer = SDL_CreateRenderer(S_Window, -1, SDL_RENDERER_ACCELERATED);
			if (S_Renderer == NULL)
			{
				SDL_Log("Renderer could not be created! SDL_Error: %s\n", SDL_GetError());
				success = false;
			}
			else
			{
				//Initialize renderer's color to black
				SDL_SetRenderDrawColor(S_Renderer, 0, 0, 0, 0xFF);
			}
		}
	}
	return success;
}

//Initializes the SDL image library used for loading textures
int init_IMG() {
	int flags = IMG_INIT_JPG | IMG_INIT_PNG;
	int init = IMG_Init(flags);

	if (init & (flags != flags)) {
		SDL_Log("Failed to init jpg and png\n");
		SDL_Log("%s\n", IMG_GetError());
		return false;
	}
	else {
		SDL_Log("Succesfully loaded IMG\n");
		return true;
	}
}

//Destroys graphics window and renderer
//Quits SDL subsystems
void close_graphics()
{
	//Destroy window and renderer
	SDL_DestroyRenderer(S_Renderer);
	SDL_DestroyWindow(S_Window);
	S_Renderer = NULL;
	S_Window = NULL;

	//Quit SDL subsystems
	SDL_Quit();

	//Quit the image loading library
	IMG_Quit();
}

//Convert r,g,b values from range [0, 1] to [0, 255]
//Sets render draw color to new r, g, b values
void set_rgb(double r, double g, double b)
{
	r *= 255;
	g *= 255;
	b *= 255;

	SDL_SetRenderDrawColor(S_Renderer, r, g, b, 0xFF);
}

void save_image_to_file(const char* filename, int w, int h)
{
	sshot = SDL_CreateRGBSurface(0, w, h, 32, rmask, gmask, bmask, amask);
	SDL_RenderReadPixels(S_Renderer, NULL, SDL_PIXELFORMAT_ARGB8888, sshot->pixels, sshot->pitch);
	SDL_SaveBMP(sshot, filename);
	SDL_FreeSurface(sshot);
}

//Return a pixel on a surface based on the x- and y location of the pixel
uint32_t get_pixel(SDL_Surface* surface, int x, int y)
{
	int bpp = surface->format->BytesPerPixel;

	//Address of the pixel we are retrieving
	uint8_t* p = (uint8_t*)surface->pixels + y * surface->pitch + x * bpp;

	switch (bpp)
	{
	case 1:
		return *p;
		break;

	case 2:
		return *(uint16_t*)p;
		break;

	case 3:
		if (SDL_BYTEORDER == SDL_BIG_ENDIAN)
		{
			return p[0] << 16 | p[1] << 8 | p[2];
		}
		else
		{
			return p[0] | p[1] << 8 | p[2] << 16;
		}
		break;

	case 4:
		return *(uint32_t*)p;

	default:
		return 0;
	}
}

//Gets rgb from texture map
void get_rgb(SDL_Surface* texture, int index_At, int index_Bt, int index_Ct, double uv[2], double rgb[3])
{

	double At[2], Bt[2], Ct[2];
	At[0] = u[index_At]; At[1] = v[index_At];
	Bt[0] = u[index_Bt]; Bt[1] = v[index_Bt];
	Ct[0] = u[index_Ct]; Ct[1] = v[index_Ct];

	double width, height;

	//Find width and height of the texture
	width = (double)texture->w;
	height = (double)texture->h;

	double x, y;
	int xi, yi;

	//Find where in the triangle we intersected (Range [0, 1]) 
	x = (1 - uv[0] - uv[1]) * At[0] + uv[0] * Bt[0] + uv[1] * Ct[0];
	y = (1 - uv[0] - uv[1]) * At[1] + uv[0] * Bt[1] + uv[1] * Ct[1];

	//IMG has (0,0) in top left corner, we want (0,0) in bottom left corner
	y = 1 - y;

	x *= width; y *= height;

	xi = (int)x; yi = (int)y;

	uint8_t r, g, b;

	uint32_t pixel = get_pixel(texture, xi, yi);
	SDL_GetRGB(pixel, texture->format, &r, &g, &b);

	rgb[0] = (r * 1.0) / 255;
	rgb[1] = (g * 1.0) / 255;
	rgb[2] = (b * 1.0) / 255;
}