#ifndef _kernel_h
#define _kernel_h

extern "C" void mat_mult_device(
	double* X, double* Y, double* Z,
	double m[4][4],
	double* x, double* y, double* z,
	int numpoints);

extern "C" int intersect_all_triangles_device(double S[3], double E[3],
	double uvt[3], double point[3], double normal[3], double obinv[4][4]);

#endif