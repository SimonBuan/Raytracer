#ifndef _obj_reader_h
#define _obj_reader_h

#include <SDL.h>

typedef struct Material {
    int index;
    char name[_MAX_FNAME];
    double Ka[3]; //ambient rgb
    double Kd[3]; //diffuse rgb
    double Ks[3]; //specular rgb
    double Ke[3];
    int illum; //illumination model, ranging from 1-10
    double Ns; //shininess of material
    double Tr;
    double Tf[3];
    double Ni;
    double d; //transperency of material
    SDL_Surface* map_Ka; //ambient texture map
    SDL_Surface* map_Kd; //diffuse texture map
    SDL_Surface* map_Ks; //specular texture map
} Material;

typedef struct Triangle {
    int A; //index of 1st point
    int B; //index of 2nd point
    int C; //index of 3rd point

    int At; //index of texture vertex of 1st point
    int Bt; //index of texture vertex of 2nd point
    int Ct; //index of texture vertex of 3rd point

    int An; //index of 1st normal
    int Bn; //index of 2nd normal
    int Cn; //index of 3rd normal

    Material mtl; //Material of the triangle
} Triangle;

extern int num_tris, num_mats, num_vn, num_v;

extern double* x, * y, * z, * xnormal, * ynormal, * znormal;
extern double* u, * v;
extern Triangle* tris;
extern Material* mats;

//Parses .obj object file and saves it to memory
int read_obj_file(const char* fname);

//Frees all memory used by the current object
void close_object();

#endif