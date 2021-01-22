#ifndef _M3d_matrix_tools_h
#define _M3d_matrix_tools_h

int M3d_print_mat (double a[4][4]);

// a = b
int M3d_copy_mat (double a[4][4], double b[4][4]);

// a = I
int M3d_make_identity (double a[4][4]);

int M3d_make_translation (double a[4][4], double dx, double dy, double dz);

int M3d_make_scaling (double a[4][4], double sx, double sy, double sz);

//Assumes cosine and sine are known
int M3d_make_x_rotation_cs (double a[4][4], double cs, double sn);

//Assumes cosine and sine are known
int M3d_make_y_rotation_cs (double a[4][4], double cs, double sn);

//Assumes cosine and sine are known
int M3d_make_z_rotation_cs (double a[4][4], double cs, double sn);

// res = a * b
int M3d_mat_mult (double res[4][4], double a[4][4], double b[4][4]);

// P = m*Q
int M3d_mat_mult_pt (double P[3],   double m[4][4], double Q[3]);


// |X0 X1 X2 ...|       |x0 x1 x2 ...|
// |Y0 Y1 Y2 ...| = m * |y0 y1 y2 ...|
// |Z0 Z1 Z2 ...|       |z0 z1 z2 ...|  
// | 1  1  1 ...|       | 1  1  1 ...|
int M3d_mat_mult_points (double *X, double *Y, double *Z,
                         double m[4][4],
                         double *x, double *y, double *z, int numpoints);

// res = a x b  , cross product of two vectors
int M3d_x_product (double res[3], double a[3], double b[3]);

//returns dot product of vectors a and b
double M3d_dot_product(double a[3], double b[3]);

//Puts a vector with length 1 and same direction as a in res
int M3d_norm(double res[3], double a[3]);

  //            |A[0] B[0]|
  //Returns det |A[1] B[1]|
double M3d_det_2x2(double A[2], double B[2]);

  //            |A[0] B[0] C[0]|
  //Returns det |A[1] B[1] C[1]|
  //  
double M3d_det_3x3(double A[3], double B[3], double C[3]);


//Macros used for creating movement sequence matrix
#define SX 0
#define SY 1
#define SZ 2

#define RX 3
#define RY 4
#define RZ 5

#define TX 6
#define TY 7
#define TZ 8

#define NX 9
#define NY 10
#define NZ 11

// create a matrix (mat) and its inverse (inv)
// that specify a sequence of movements....
// movement_type_list[k] is an integer that
// specifies the type of matrix to be used in the
// the k-th movement.  the parameter that each
// matrix needs is supplied in parameter_list[k].
// return 1 if successful, 0 if error
int M3d_make_movement_sequence_matrix (
                              double mat[4][4],
                              double inv[4][4],
                              int num_movements,
                              int *movement_type_list,
                              double *parameter_list );


// Construct the view matrix and its inverse given the location
// of the eye, the center of interest, and an up point.
// return 1 if successful, 0 otherwise.                   
int M3d_view (double view[4][4], double view_inverse[4][4],
              double eye[3], double coi[3], double up[3]);

#endif 