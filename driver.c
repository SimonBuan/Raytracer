#include <stdint.h>

#include <SDL.h>
#include <SDL_image.h>

#include "M3d_matrix_tools.c"
#include "light_model.c"
#include "obj_reader.c"

//Boolean constants
const int true = 1;
const int false = 0;

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
const int SCREEN_WIDTH = 400;
const int SCREEN_HEIGHT = 400;


Triangle tris_o[100000];


//Initializes SDL, and the graphics window and renderer
//Sets render draw color to black
//Returns 1 on success, 0 on failure
int init_graphics(int w, int h)
{
	SDL_Log("Initalizing graphics\n");
	int success = true;

	//Initialize SDL
	if(SDL_Init(SDL_INIT_VIDEO) < 0)
	{
		SDL_Log("SDL could not initialize! SDL_Error: %s\n", SDL_GetError());
		success = false;
	}
	else
	{
		//Create window
		S_Window =  SDL_CreateWindow("Raytracer", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, w, h, SDL_WINDOW_SHOWN);
		if(S_Window == NULL)
		{
			SDL_Log("Window could not be created! SDL_Error: %s\n", SDL_GetError());
			success = false;
		}
		else
		{
			//Create renderer for window
			S_Renderer = SDL_CreateRenderer(S_Window, -1, SDL_RENDERER_ACCELERATED);
			if(S_Renderer == NULL)
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
int init_IMG(){
  int flags = IMG_INIT_JPG | IMG_INIT_PNG;
  int init = IMG_Init(flags);

  if(init&flags != flags){
    SDL_Log("Failed to init jpg and png\n");
    SDL_Log("%s\n", IMG_GetError());
    return false;
  }
  else{
    SDL_Log("Succesfully loaded IMG\n");
    return true;
  }
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

void center_object()
{
  // center the object at the origin
  double xc,yc,zc ;
  xc = yc = zc = 0 ;
  int i ;
  for (i = 0 ; i < num_tris ; i++) {
    xc += tris[i].A[0]; yc += tris[i].A[1]; zc += tris[i].A[2];
    xc += tris[i].B[0]; yc += tris[i].B[1]; zc += tris[i].B[2];
    xc += tris[i].C[0]; yc += tris[i].C[1]; zc += tris[i].C[2];
  }
  xc /= (3*num_tris);
  yc /= (3*num_tris);
  zc /= (3*num_tris);
  for (i = 0 ; i < num_tris ; i++) {
    tris[i].A[0] -= xc;  tris[i].A[1] -= yc;  tris[i].A[2] -= zc;
    tris[i].B[0] -= xc;  tris[i].B[1] -= yc;  tris[i].B[2] -= zc;
    tris[i].C[0] -= xc;  tris[i].C[1] -= yc;  tris[i].C[2] -= zc;
  }
}

void save_image_to_file(const void *filename, int w, int h)
{
	sshot = SDL_CreateRGBSurface(0, w, h, 32, rmask, gmask,bmask, amask);
	SDL_RenderReadPixels(S_Renderer, NULL, SDL_PIXELFORMAT_ARGB8888, sshot->pixels, sshot->pitch);
	SDL_SaveBMP(sshot, filename);
	SDL_FreeSurface(sshot);
}

double det_2x2(double A[2], double B[2]){
  //            |A[0] B[0]|
  //Returns det |A[1] B[1]|
  double det = A[0]*B[1] - A[1]*B[0];
  return det;
}

double det_3x3(double A[3], double B[3], double C[3]){
  //            |A[0] B[0] C[0]|
  //Returns det |A[1] B[1] C[1]|
  //            |A[2] B[2] C[2]|
  double det = 0;
  double tempA[2], tempB[2];
  tempA[0] = B[1]; tempA[1] = B[2];
  tempB[0] = C[1]; tempB[1] = C[2];
  det += A[0] * det_2x2(tempA, tempB);
  tempA[0] = A[1]; tempA[1] = A[2];
  tempB[0] = C[1]; tempB[1] = C[2];
  det += -B[0] * det_2x2(tempA, tempB);
  tempA[0] = A[1]; tempA[1] = A[2];
  tempB[0] = B[1]; tempB[1] = B[2];
  det += C[0] * det_2x2(tempA, tempB);
  return det;
}

double intersect_single_triangle(double S[3], double E[3], double uv[2], Triangle tri){
  double AB[3], AC[3], ES[3], AS[3];
  double A[3], B[3], C[3];
  double t;

  
  A[0] = tri.A[0]; A[1] = tri.A[1]; A[2] = tri.A[2];
  B[0] = tri.B[0]; B[1] = tri.B[1]; B[2] = tri.B[2];
  C[0] = tri.C[0]; C[1] = tri.C[1]; C[2] = tri.C[2];
  
  AB[0] = B[0] - A[0]; AB[1] = B[1] - A[1]; AB[2] = B[2] - A[2];
  AC[0] = C[0] - A[0]; AC[1] = C[1] - A[1]; AC[2] = C[2] - A[2];
  ES[0] = S[0] - E[0]; ES[1] = S[1] - E[1]; ES[2] = S[2] - E[2];
  AS[0] = S[0] - A[0]; AS[1] = S[1] - A[1]; AS[2] = S[2] - A[2];

  double den = det_3x3(AB, AC, ES);
  if(den == 0){
    return -1;
  }
  double topt, topu, topv;
  topt =  det_3x3(AB, AC, AS);
  t = topt/den;
  if(t < 0){
    return -1;
  }
  topu = det_3x3(AS, AC, ES);
  uv[0] = topu/den;
  if((uv[0] < 0) || (uv[0] > 1)){
    return -1;
  }

  topv = det_3x3(AB, AS, ES);
  uv[1] = topv/den;
  if((uv[1] < 0) || (uv[1] > 1)){
    return -1;
  }
  if(uv[0] + uv[1] > 1){
    return -1;
  }
  return t;
}

int intersect_all_triangles(double S[3], double E[3],
 double uvt[3], double point[3], double normal[3], double obinv[4][4])
// return index of closest triangle or -1 if there is none.
// and load up the arrays uvt,point,normal
{
  int i;
  int closest = -1;
  double tempUV[2], tempt;
  uvt[2] = 1e50;

  for(i = 0; i  < num_tris; i++){

    tempt = intersect_single_triangle(S, E, tempUV, tris_o[i]);

    if((tempt > 0) && (tempt < uvt[2])){
      uvt[2] = tempt;
      uvt[0] = tempUV[0];
      uvt[1] = tempUV[1];
      closest = i;
    }
  }

  if(closest != -1){
  	//Load point with coordinates of intersection between ray and object
    point[0] = S[0] + uvt[2]*(E[0]-S[0]);
    point[1] = S[1] + uvt[2]*(E[1]-S[1]);
    point[2] = S[2] + uvt[2]*(E[2]-S[2]);

    interpolate_normal_vector (normal, tris[closest].An, tris[closest].Bn, tris[closest].Cn, uvt, obinv);
	}
  return closest;
}

//Return a pixel on a surface based on the x- and y location of the pixel
uint32_t get_pixel(SDL_Surface *surface, int x, int y)
{
  int bpp = surface->format->BytesPerPixel;

  //Address of the pixel we are retrieving
  uint8_t *p = (uint8_t *)surface->pixels + y * surface->pitch + x * bpp;

  switch(bpp)
  {
    case 1:
      return *p;
      break;

    case 2:
      return *(uint16_t *)p;
      break;

    case 3:
      if(SDL_BYTEORDER == SDL_BIG_ENDIAN)
      {
        return p[0] << 16 | p[1] << 8 | p[2];
      }
      else
      {
        return p[0] | p[1] << 8 | p[2] << 16;
      }
      break;

    case 4:
      return *(uint32_t *)p;

    default:
      return 0;
  }
}

//Gets rgb from texture map
void get_rgb(SDL_Surface *texture, double At[2], double Bt[2], double Ct[2], double uv[2], double rgb[3])
{
  

  double A[2], B[2], C[2];
  double width, height;
  
  //Find width and height of the texture
  width = (double) texture->w;
  height = (double) texture->h;

  double x,y;
  int xi, yi;

  //Find where in the triangle we intersected (Range [0, 1]) 
  x = (1-uv[0]-uv[1])*At[0] + uv[0]*Bt[0] + uv[1]*Ct[0];
  y = (1-uv[0]-uv[1])*At[1] + uv[0]*Bt[1] + uv[1]*Ct[1];

  y = 1 - y;

  x *= width; y *= height;

  xi = (int) x; yi = (int) y;

  uint8_t r, g, b;

  uint32_t pixel = get_pixel(texture, xi, yi);
  SDL_GetRGB(pixel, texture->format, &r, &g, &b);

  rgb[0] = (r*1.0)/255;
  rgb[1] = (g*1.0)/255;
  rgb[2] = (b*1.0)/255;
}

int main(int argc, char **argv)
{
  init_graphics(SCREEN_WIDTH,SCREEN_HEIGHT);
  init_IMG();

  double degrees_of_half_angle ;

  degrees_of_half_angle = 30 ;
  read_obj_file("./objects/santa/12165_Santa_Claus_v1_l2.obj");

  center_object() ;

  double tan_half = tan(degrees_of_half_angle*M_PI/180);

  

  int s,e,frame_number ;
  s = 0 ; e = 60 ;
  for(frame_number = s; frame_number <= e; frame_number++){
    SDL_Log("frame: %d\n", frame_number);
    set_rgb(0.5,0.5,0.5);
    SDL_RenderClear(S_Renderer);

    double eangle = 2*M_PI*frame_number/e ;
    double eye[3] ;

    //Location of eye (world space)
    eye[0] =   75.0*cos(eangle) ;
    eye[1] =   30.0;
    eye[2] =   75.0*sin(eangle) ;

    double coi[3] ;
    
    //Center of interest/where the eye is looking (world space)
    coi[0] =  0 ;
    coi[1] =  0 ;
    coi[2] =  0 ;

    double up[3] ;

    //What direction is up for the eye
    up[0] = eye[0] ;
    up[1] = eye[1] + 1 ;
    up[2] = eye[2] ;

    double vm[4][4], vi[4][4];
    M3d_view(vm,vi,  eye,coi,up) ;

    double light_in_world_space[3] ;
    light_in_world_space[0] =    300 ;
    light_in_world_space[1] =    150 ;
    light_in_world_space[2] =    300 ;
    M3d_mat_mult_pt(light_in_eye_space, vm, light_in_world_space) ;

    double Ka[3], Kd[3], Ks[3];

    // Transform the object :
    double Tvlist[100];
    int Tn, Ttypelist[100];
    double m[4][4], mi[4][4];
    double obmat[4][4], obinv[4][4];

    Tn = 0 ;
    Ttypelist[Tn] = RX ; Tvlist[Tn] = 270 ; Tn++ ;  

    M3d_make_movement_sequence_matrix(m, mi, Tn, Ttypelist, Tvlist);
    M3d_mat_mult(obmat, vm, m) ;
    M3d_mat_mult(obinv, mi, vi) ;
    
    int k ;
    for (k = 0 ; k < num_tris ; k++) {
      M3d_mat_mult_pt(tris_o[k].A, obmat,  tris[k].A) ;
      M3d_mat_mult_pt(tris_o[k].B, obmat,  tris[k].B) ;
      M3d_mat_mult_pt(tris_o[k].C, obmat,  tris[k].C) ;
    }


    ///////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////
    double origin[3], screen_pt[3] ;
    double uvt[3], point[3], normal[3] ;
    double argb[3] ;

    origin[0] = 0;
    origin[1] = 0;
    origin[2] = 0;

    int x_pix, y_pix;
    for(x_pix = 0; x_pix < SCREEN_WIDTH; x_pix++)
    {
      SDL_Log("%d/%d\n", x_pix, SCREEN_WIDTH);
      for(y_pix = 0; y_pix < SCREEN_HEIGHT; y_pix++)
      {
        screen_pt[0] = x_pix - SCREEN_WIDTH/2;
        screen_pt[1] = y_pix - SCREEN_HEIGHT/2;
        screen_pt[2] = (SCREEN_WIDTH/2) / tan_half;

        s = intersect_all_triangles(origin, screen_pt, uvt, point, normal, obinv) ;
        if (s == -1) 
        {
         argb[0] = argb[1] = argb[2] = 0 ;
        } 
       else 
       {
          if(tris[s].mtl.index != -1)
          {
            if(tris[s].mtl.map_Ka)
            {
              get_rgb(tris[s].mtl.map_Ka, tris[s].At, tris[s].Bt, tris[s].Ct, uvt, Ka);
            }
            else{
              Ka[0] = tris[s].mtl.Ka[0];
              Ka[1] = tris[s].mtl.Ka[1];
              Ka[2] = tris[s].mtl.Ka[2];
            }

            if(tris[s].mtl.map_Kd)
            {
              get_rgb(tris[s].mtl.map_Kd, tris[s].At, tris[s].Bt, tris[s].Ct, uvt, Kd);
            }
            else{
              Kd[0] = tris[s].mtl.Kd[0];
              Kd[1] = tris[s].mtl.Kd[1];
              Kd[2] = tris[s].mtl.Kd[2];
            }

            if(tris[s].mtl.map_Ks)
            {
              get_rgb(tris[s].mtl.map_Ks, tris[s].At, tris[s].Bt, tris[s].Ct, uvt, Ks);
            }
            else{
              Ks[0] = tris[s].mtl.Ks[0];
              Ks[1] = tris[s].mtl.Ks[1];
              Ks[2] = tris[s].mtl.Ks[2];
            }
        }
        else
        {
          Ka[0] = 1.0; Ka[1] = 1.0; Ka[2] = 1.0;
          Kd[0] = 1.0; Kd[1] = 1.0; Kd[2] = 1.0;
          Ks[0] = 1.0; Ks[1] = 1.0; Ks[2] = 1.0;
        }
        Light_Model (Ka, Kd, Ks, origin, point, normal, argb);
        
      }	

      int screen_y;
      screen_y = SCREEN_HEIGHT - y_pix;

      set_rgb(argb[0], argb[1], argb[2]);
      SDL_RenderDrawPoint(S_Renderer, x_pix, screen_y);

      } // end for y_pix
    } // end for x_pix


    SDL_RenderPresent(S_Renderer) ;
    char fname[200] ;
    sprintf(fname, "pic/pic%04d.bmp",frame_number) ;
    save_image_to_file(fname, SCREEN_WIDTH, SCREEN_HEIGHT) ;
  } // end for frame_number
}
