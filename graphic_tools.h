#ifndef _graphic_tools_h
#define _graphic_tools_h

#include <SDL.h>

//Window renderer
extern SDL_Renderer* S_Renderer;

//Graphics window dimensions
extern const int SCREEN_WIDTH;
extern const int SCREEN_HEIGHT;

//Initializes SDL, and the graphics window and renderer
//Sets render draw color to black
//Returns 1 on success, 0 on failure
int init_graphics();

//Initializes the SDL image library used for loading textures
int init_IMG();

//Destroys graphics window and renderer
//Quits SDL subsystems
void close_graphics();

//Convert r,g,b values from range [0, 1] to [0, 255]
//Sets render draw color to new r, g, b values
void set_rgb(double r, double g, double b);

//Saves the current rendering to a bitmap file
void save_image_to_file(const char* filename);

//Return a pixel on a surface based on the x- and y location of the pixel
uint32_t get_pixel(SDL_Surface* surface, int x, int y);

//Gets rgb from texture map
void get_rgb(SDL_Surface* texture, int index_At, int index_Bt, int index_Ct, double uv[2], double rgb[3]);

#endif