#ifndef _kernel_h
#define _kernel_h

#include "obj_reader.h"

#ifdef __cplusplus
extern "C" {
#endif

void mat_mult_device(
		double* X, double* Y, double* Z,
		double m[4][4],
		double* x, double* y, double* z,
		int numpoints);

int intersect_all_triangles_device(double S[3], double E[3],
		double uv[2], double point[3], Triangle* tris, double* x, double* y, double* z, int num_tris);

#ifdef __cplusplus
}
#endif


#endif