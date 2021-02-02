#include "M3d_matrix_tools.h"
#include "obj_reader.h"
#include "light_model.h"

double intersect_single_triangle(double S[3], double E[3], double uv[2], Triangle tri){
  double AB[3], AC[3], ES[3], AS[3];
  double A[3], B[3], C[3];
  double t;

  //Get the indeces of the vertices in the triangle
  int index_A = tri.A;
  int index_B = tri.B;
  int index_C = tri.C;

  //Get the positions of the vertices from corresponding index
  A[0] = x[index_A]; A[1] = y[index_A]; A[2] = z[index_A];
  B[0] = x[index_B]; B[1] = y[index_B]; B[2] = z[index_B];
  C[0] = x[index_C]; C[1] = y[index_C]; C[2] = z[index_C];

  //Find vectors between points in triangle, 
  //as well as with start and end of ray
  AB[0] = B[0] - A[0]; AB[1] = B[1] - A[1]; AB[2] = B[2] - A[2];
  AC[0] = C[0] - A[0]; AC[1] = C[1] - A[1]; AC[2] = C[2] - A[2];
  ES[0] = S[0] - E[0]; ES[1] = S[1] - E[1]; ES[2] = S[2] - E[2];
  AS[0] = S[0] - A[0]; AS[1] = S[1] - A[1]; AS[2] = S[2] - A[2];

  double den = M3d_det_3x3(AB, AC, ES);
  if(den == 0){
    return -1;
  }
  double topt, topu, topv;
  topt =  M3d_det_3x3(AB, AC, AS);
  t = topt/den;
  if(t < 0){
    return -1;
  }
  topu = M3d_det_3x3(AS, AC, ES);
  uv[0] = topu/den;
  if((uv[0] < 0) || (uv[0] > 1)){
    return -1;
  }

  topv = M3d_det_3x3(AB, AS, ES);
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
    //Find the distance between start of ray and triangle-intersection point
    tempt = intersect_single_triangle(S, E, tempUV, tris[i]);

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


    //Find normal vector information at the closest triangle
    double An[3], Bn[3], Cn[3];
    int index_A = tris[closest].An;
    int index_B = tris[closest].Bn;
    int index_C = tris[closest].Cn;

    An[0] = xnormal[index_A];
    An[1] = ynormal[index_A];
    An[2] = znormal[index_A];

    Bn[0] = xnormal[index_B];
    Bn[1] = ynormal[index_B];
    Bn[2] = znormal[index_B];

    Cn[0] = xnormal[index_C];
    Cn[1] = ynormal[index_C];
    Cn[2] = znormal[index_C];

    //Interpolate between the normal at each vertex to find normal at intersection point
    interpolate_normal_vector (normal, An, Bn, Cn, uvt, obinv);
	}
  return closest;
}