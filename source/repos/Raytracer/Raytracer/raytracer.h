#ifndef _raytracer_h
#define _raytracer_h

#include "obj_reader.h"

//Intersects a ray (S->E) with tri.
//Loads uv with appropraite uv-values.
double intersect_single_triangle(double S[3], double E[3], double uv[2], Triangle tri);

//Intersects all triangles with a ray (S->E) 
//and finds the closest intersection to the start of the ray (can't be behind start of ray).
// return index of closest triangle or -1 if there is none.
//Loads uvt with uv-values, as well as t indicating the  disctance from start of ray
//Loads point with xyz-coordinates of intersection point
//Loads normal with the normal vector of the triangle at intersection
//Obinv is the inverse of the matrix used to transform object from object-space to eye-space.
int intersect_all_triangles(double S[3], double E[3],
    double uvt[3], double point[3], double normal[3], double obinv[4][4]);
#endif

