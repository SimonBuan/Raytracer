#ifndef _light_model_h
#define _light_model_h

extern double light_in_eye_space[];

// s,p,n in eyespace
// Ka, Kd, Ks == inherent ambient, diffuse and spec color of object (input to this function)
// s = location of start of ray (probably the eye)
// p = point on object (input to this function)
// n = normal to the object at p (input to this function)
// argb == actual color of object (output of this function)
// globals : AMBIENT, MAX_DIFFUSE, SPECPOW, light_in_eye_space[3]
// return 1 if successful, 0 if error
int Light_Model(double Ka[3],
	double Kd[3],
	double Ks[3],
	double s[3],
	double p[3],
	double n[3],
	double argb[3]);

//Interpolate normal vector based on uv-values
//tri is the index of the triangle where intersection happened
//resulting vector is stored in normal
void interpolate_normal_vector(int tri, double uv[2], double obinv[4][4], double normal[3]);


#endif
