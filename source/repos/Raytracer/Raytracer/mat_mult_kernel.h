#ifndef _mat_mult_kernel_h
#define _mat_mult_kernel_h

extern "C" void mat_mult_device(
	double* X, double* Y, double* Z,
	double m[4][4],
	double* x, double* y, double* z,
	int numpoints);

#endif