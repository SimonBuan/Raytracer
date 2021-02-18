#include "M3d_matrix_tools.h"
#include <math.h>
#include "kernel.h"
#include "obj_reader.h"
#include <stdio.h>

// To support the light model :
int num_lights;
double light_in_eye_space[100][3];
double light_rgb[100][3];
double light_attenuation[100][2];
double light_max_dist[100];
double AMBIENT = 0.2;
double DIFFUSE = 0.5;
double SPECPOW = 50;

int Light_Model_Single(double Ka[3],
	double Kd[3],
	double Ks[3],
	double s[3],
	double p[3],
	double n[3],
	double argb[3],
	int light_num)
	// s,p,n in eyespace

	// Ka, Kd, Ks == inherent ambient, diffuse and spec color of object (input to this function)
	// s = location of start of ray (probably the eye)
	// p = point on object (input to this function)
	// n = normal to the object at p (input to this function)
	// argb == actual color of object (output of this function)
	// globals : AMBIENT, MAX_DIFFUSE, SPECPOW, light_in_eye_space[3]

	// return 1 if successful, 0 if error
{
	double len;
	double N[3]; //Normal vector
	len = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
	if (len == 0) return 0;
	if (M3d_norm(N, n) == 0) return 0;

	double L[3]; //Vector from point to light
	L[0] = light_in_eye_space[light_num][0] - p[0];
	L[1] = light_in_eye_space[light_num][1] - p[1];
	L[2] = light_in_eye_space[light_num][2] - p[2];
	len = sqrt(L[0] * L[0] + L[1] * L[1] + L[2] * L[2]);
	if (len == 0) return 0;
	if (M3d_norm(L, L) == 0) return 0;
	double NdotL = M3d_dot_product(N, L);

	if (len > light_max_dist[light_num]) {
		argb[0] = 0;
		argb[1] = 0;
		argb[2] = 0;
		return 1;
	}
	double attenuation = 1 + light_attenuation[light_num][0] * len + light_attenuation[light_num][1] * len * len;
	

	double shadow_S[3]; //Start of ray used for checking shadows
	shadow_S[0] = p[0] + L[0] * 0.001;
	shadow_S[1] = p[1] + L[1] * 0.001;
	shadow_S[2] = p[2] + L[2] * 0.001;

	//Arrays not actually used, needed for function call
	double uv[2];
	double point[3];

	//Check if there is an object between light and our intersection point
	if (intersect_all_triangles_device(shadow_S, light_in_eye_space[light_num], uv, point) != -1) {
		//Light blocked by object, just want ambient
		argb[0] = 0;
		argb[1] = 0;
		argb[2] = 0;
		return 1;
	}

	double V[3]; //Vector from camera to point
	V[0] = s[0] - p[0];
	V[1] = s[1] - p[1];
	V[2] = s[2] - p[2];
	if (M3d_norm(V, V) == 0) return 0;

	

	/////DIFFUSE/////

	double diffuse[3];
	diffuse[0] = NdotL * Kd[0] * DIFFUSE * light_rgb[light_num][0];
	diffuse[1] = NdotL * Kd[1] * DIFFUSE * light_rgb[light_num][1];
	diffuse[2] = NdotL * Kd[2] * DIFFUSE * light_rgb[light_num][2];

	for (int i = 0; i < 3; i++)
	{
		if (diffuse[i] < 0) diffuse[i] = 0;
	}

	/////SPECULAR/////
	double f;

	//Phong's Specular Term
	double R[3];
	R[0] = 2 * NdotL * N[0] - L[0];
	R[1] = 2 * NdotL * N[1] - L[1];
	R[2] = 2 * NdotL * N[2] - L[2];

	double RdotV = M3d_dot_product(R, V);
	if (RdotV > 0) f = pow(RdotV, SPECPOW);
	else f = 0;

	double specular[3];
	specular[0] = f * Ks[0] * (1.0 - DIFFUSE - AMBIENT) * light_rgb[light_num][0];
	specular[1] = f * Ks[1] * (1.0 - DIFFUSE - AMBIENT) * light_rgb[light_num][1];
	specular[2] = f * Ks[2] * (1.0 - DIFFUSE - AMBIENT) * light_rgb[light_num][2];

	/////FINAL COLOR/////
	argb[0] = (diffuse[0] + specular[0]) / attenuation;
	argb[1] = (diffuse[1] + specular[1]) / attenuation;
	argb[2] = (diffuse[2] + specular[2]) / attenuation;

	return 1;
}

int Light_Model(double Ka[3],
		double Kd[3],
		double Ks[3],
		double s[3],
		double p[3],
		double n[3],
		double argb[3])
{
	/////AMBIENT/////
	double ambient[3];
	ambient[0] = AMBIENT * Ka[0];
	ambient[1] = AMBIENT * Ka[1];
	ambient[2] = AMBIENT * Ka[2];
	int i, j;
	double rgb[3];
	argb[0] = ambient[0]; argb[1] = ambient[1]; argb[2] = ambient[2];
	for (i = 0; i < num_lights; i++) {
		
		Light_Model_Single(Ka, Kd, Ks, s, p, n, rgb, i);
		for (j = 0; j < 3; j++) {
			argb[j] += rgb[j];
		}
	}
	for (j = 0; j < 3; j++) {
		if (argb[j] > 1) argb[j] = 1;
	}
	return 1;
}

//Interpolate normal vector based on uv-values
//tri is the index of the triangle where intersection happened
//resulting vector is stored in normal
void interpolate_normal_vector(int tri, double uv[2], double obinv[4][4], double normal[3]) {
	//Find normal vector information at the closest triangle
	double An[3], Bn[3], Cn[3];
	int index_A = tris[tri].An;
	int index_B = tris[tri].Bn;
	int index_C = tris[tri].Cn;

	An[0] = xnormal[index_A];
	An[1] = ynormal[index_A];
	An[2] = znormal[index_A];

	Bn[0] = xnormal[index_B];
	Bn[1] = ynormal[index_B];
	Bn[2] = znormal[index_B];

	Cn[0] = xnormal[index_C];
	Cn[1] = ynormal[index_C];
	Cn[2] = znormal[index_C];

	double Ntemp[3];

	//Interpolating between the 3 normal vectors
	Ntemp[0] = (1 - uv[0] - uv[1]) * An[0] + uv[0] * Bn[0] + uv[1] * Cn[0];
	Ntemp[1] = (1 - uv[0] - uv[1]) * An[1] + uv[0] * Bn[1] + uv[1] * Cn[1];
	Ntemp[2] = (1 - uv[0] - uv[1]) * An[2] + uv[0] * Bn[2] + uv[1] * Cn[2];

	//Transforming normal to eye space
	normal[0] = Ntemp[0] * obinv[0][0] + Ntemp[1] * obinv[1][0] + Ntemp[2] * obinv[2][0];
	normal[1] = Ntemp[0] * obinv[0][1] + Ntemp[1] * obinv[1][1] + Ntemp[2] * obinv[2][1];
	normal[2] = Ntemp[0] * obinv[0][2] + Ntemp[1] * obinv[1][2] + Ntemp[2] * obinv[2][2];
}
