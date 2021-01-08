#include <stdio.h>
#include <math.h>
#include "M3d_matrix_tools.h"

/*

 ( x')          (x)
 ( y')  =   M * (y)  
 ( z')          (z)
 ( 1 )          (1)

instead of (x',y',z',1) = (x,y,z,1) * M  

*/




int M3d_print_mat (double a[4][4])
{
  int r,c ;
  for (r = 0 ; r < 4 ; r++ ) {
      for (c = 0 ; c < 4 ; c++ ) {
           printf(" %12.4lf ",a[r][c]) ;
      }
      printf("\n") ;
  }

  return 1 ;
} 





int M3d_copy_mat (double a[4][4], double b[4][4])
// a = b
{
  int r,c ;
  for (r = 0 ; r < 4 ; r++ ) {
      for (c = 0 ; c < 4 ; c++ ) {
           a[r][c] = b[r][c] ;
      }
  }

  return 1 ;
} 





int M3d_make_identity (double a[4][4])
// a = I
{
  int r,c ;
  for (r = 0 ; r < 4 ; r++ ) {
      for (c = 0 ; c < 4 ; c++ ) {
           if (r == c) a[r][c] = 1.0 ;
               else    a[r][c] = 0.0 ;
      }
  }

  return 1 ;
} 





int M3d_make_translation (double a[4][4], double dx, double dy, double dz)
{
  M3d_make_identity(a) ;
  a[0][3] =  dx ;  a[1][3] = dy ;  a[2][3] = dz ;
  return 1 ;
}





int M3d_make_scaling (double a[4][4], double sx, double sy, double sz)
{
  M3d_make_identity(a) ;
  a[0][0] =  sx ;  a[1][1] = sy ;  a[2][2] = sz ;
  return 1 ;
}

int M3d_make_x_rotation_cs (double a[4][4], double cs, double sn)
// this one assumes cosine and sine are already known
{
  M3d_make_identity(a) ;

  a[1][1] =   cs ;  a[1][2] = -sn ;
  a[2][1] =   sn ;  a[2][2] =  cs ;

  return 1 ;
}



int M3d_make_y_rotation_cs (double a[4][4], double cs, double sn)
// this one assumes cosine and sine are already known
{
  M3d_make_identity(a) ;

  a[0][0] =   cs ;  a[0][2] =  sn ;
  a[2][0] =  -sn ;  a[2][2] =  cs ;

  return 1 ;
}


int M3d_make_z_rotation_cs (double a[4][4], double cs, double sn)
// this one assumes cosine and sine are already known
{
  M3d_make_identity(a) ;

  a[0][0] =   cs ;  a[0][1] = -sn ;
  a[1][0] =   sn ;  a[1][1] =  cs ;

  return 1 ;
}





int M3d_mat_mult (double res[4][4], double a[4][4], double b[4][4])
// res = a * b
// this is SAFE, i.e. the user can make a call such as 
// M3d_mat_mult(p,  p,q) or M3d_mat_mult(p,  q,p) or  M3d_mat_mult(p, p,p)
{
  double sum ;
  int k ;
  int r,c ;
  double tmp[4][4] ;

  for (r = 0 ; r < 4 ; r++ ) {
      for (c = 0 ; c < 4 ; c++ ) {
           sum = 0.0 ;
           for (k = 0 ; k < 4 ; k++) {
                 sum = sum + a[r][k]*b[k][c] ;
           }
           tmp[r][c] = sum ;
      }
  }


  M3d_copy_mat (res,tmp) ;

  return 1 ;
}





int M3d_mat_mult_pt (double P[3],   double m[4][4], double Q[3])
// P = m*Q
// SAFE, user may make a call like M3d_mat_mult_pt (W, m,W) ;
{
  double u,v,t ;

  u = m[0][0]*Q[0] + m[0][1]*Q[1] + m[0][2]*Q[2] + m[0][3] ;
  v = m[1][0]*Q[0] + m[1][1]*Q[1] + m[1][2]*Q[2] + m[1][3] ;
  t = m[2][0]*Q[0] + m[2][1]*Q[1] + m[2][2]*Q[2] + m[2][3] ;  

  P[0] = u ;
  P[1] = v ;
  P[2] = t ;
  
  return 1 ;
}





int M3d_mat_mult_points (double *X, double *Y, double *Z,
                         double m[4][4],
                         double *x, double *y, double *z, int numpoints)
// |X0 X1 X2 ...|       |x0 x1 x2 ...|
// |Y0 Y1 Y2 ...| = m * |y0 y1 y2 ...|
// |Z0 Z1 Z2 ...|       |z0 z1 z2 ...|  
// | 1  1  1 ...|       | 1  1  1 ...|

// SAFE, user may make a call like M3d_mat_mult_points (x,y,z,  m, x,y,z,  n) ;
{
  double u,v,t ;
  int i ;

  for (i = 0 ; i < numpoints ; i++) {
    u = m[0][0]*x[i] + m[0][1]*y[i] + m[0][2]*z[i] + m[0][3] ;
    v = m[1][0]*x[i] + m[1][1]*y[i] + m[1][2]*z[i] + m[1][3] ;
    t = m[2][0]*x[i] + m[2][1]*y[i] + m[2][2]*z[i] + m[2][3] ;    

    X[i] = u ;
    Y[i] = v ;
    Z[i] = t ;
  }

  return 1 ;
}

int M3d_x_product (double res[3], double a[3], double b[3])
// res = a x b  , cross product of two vectors
// SAFE: it is ok to make a call such as
// M3d_x_product (a,  a,b) or
// M3d_x_product (b,  a,b) or
// M3d_x_product (a,  a,a) 
{
    double r[3] ;
    int v ;
    
    r[0] = a[1]*b[2] - b[1]*a[2] ;
    r[1] = b[0]*a[2] - a[0]*b[2] ;
    r[2] = a[0]*b[1] - b[0]*a[1] ;

    res[0] = r[0] ;
    res[1] = r[1] ;
    res[2] = r[2] ;

    if ((res[0] == 0) && (res[1] == 0) && (res[2] == 0)) {
	v = 0 ;
    } else {
	v = 1 ;
    }

    return v ;
}

double M3d_dot_product(double a[3], double b[3])
//returns dot product of vectors a and b
{
        return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

int M3d_norm(double res[3], double a[3]){
        //Puts a vector with length 1 and same direction as a in res
        double mag = sqrt(M3d_dot_product(a, a));
        
        if(mag == 0) return 0;

        res[0] = a[0]/mag;
        res[1] = a[1]/mag;
        res[2] = a[2]/mag;

        return 1;
}

double M3d_det_2x2(double A[2], double B[2]){
  //            |A[0] B[0]|
  //Returns det |A[1] B[1]|
  double det = A[0]*B[1] - A[1]*B[0];
  return det;
}

double M3d_det_3x3(double A[3], double B[3], double C[3]){
  //            |A[0] B[0] C[0]|
  //Returns det |A[1] B[1] C[1]|
  //            |A[2] B[2] C[2]|
  double det = 0;
  double tempA[2], tempB[2];
  tempA[0] = B[1]; tempA[1] = B[2];
  tempB[0] = C[1]; tempB[1] = C[2];
  det += A[0] * M3d_det_2x2(tempA, tempB);
  tempA[0] = A[1]; tempA[1] = A[2];
  tempB[0] = C[1]; tempB[1] = C[2];
  det += -B[0] * M3d_det_2x2(tempA, tempB);
  tempA[0] = A[1]; tempA[1] = A[2];
  tempB[0] = B[1]; tempB[1] = B[2];
  det += C[0] * M3d_det_2x2(tempA, tempB);
  return det;
}





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


  

int M3d_make_movement_sequence_matrix (
                              double mat[4][4],
                              double inv[4][4],
                              int num_movements,
                              int *movement_type_list,
                              double *parameter_list )
// create a matrix (mat) and its inverse (inv)
// that specify a sequence of movements....
// movement_type_list[k] is an integer that
// specifies the type of matrix to be used in the
// the k-th movement.  the parameter that each
// matrix needs is supplied in parameter_list[k].

// return 1 if successful, 0 if error

// the codes for movement_type_list are :
// 0 - scale x
// 1 - scale y
// 2 - scale z

// 3 - rotate x
// 4 - rotate y
// 5 - rotate z
   
// 6 - translate x
// 7 - translate y
// 8 - translate z

// 9  - negate x...reflect in the yz plane
// 10 - negate y...relfect in the xz plane
// 11 - negate z...reflect in the xy plane
{
 int i;
 int m ;
 double p ;

 double tmp[4][4] ;
 
 M3d_make_identity( mat ) ;
 M3d_make_identity( inv ) ;


 for (i = 0 ; i < num_movements ; i++) {

   m = movement_type_list[i] ;
   p = parameter_list[i] ;

   switch (m) {

   case SX : 
        M3d_make_scaling (tmp,    p, 1.0, 1.0 ) ;
	M3d_mat_mult(mat,  tmp,mat) ;

        M3d_make_scaling (tmp,    1.0/p, 1.0, 1.0 ) ;
	M3d_mat_mult(inv,  inv,tmp) ;	
        break ;

   case SY : 
        M3d_make_scaling (tmp,    1.0, p, 1.0 ) ;
	M3d_mat_mult(mat,  tmp,mat) ;

        M3d_make_scaling (tmp,    1.0, 1.0/p, 1.0 ) ;
	M3d_mat_mult(inv,  inv,tmp) ;		
        break ;

   case SZ : 
        M3d_make_scaling (tmp,    1.0, 1.0, p ) ;
	M3d_mat_mult(mat,  tmp,mat) ;

        M3d_make_scaling (tmp,    1.0, 1.0, 1.0/p ) ;
	M3d_mat_mult(inv,  inv,tmp) ;			
        break ;



	
   case RX :
        M3d_make_x_rotation_cs (tmp, cos(p*M_PI/180), sin(p*M_PI/180)) ;
	M3d_mat_mult(mat,  tmp,mat) ;

        M3d_make_x_rotation_cs (tmp, cos(-p*M_PI/180), sin(-p*M_PI/180)) ;
	M3d_mat_mult(inv,  inv,tmp) ;			
        break ;

   case RY :
        M3d_make_y_rotation_cs (tmp, cos(p*M_PI/180), sin(p*M_PI/180)) ;
	M3d_mat_mult(mat,  tmp,mat) ;

        M3d_make_y_rotation_cs (tmp, cos(-p*M_PI/180), sin(-p*M_PI/180)) ;
	M3d_mat_mult(inv,  inv,tmp) ;				
        break ;

   case RZ :
        M3d_make_z_rotation_cs (tmp, cos(p*M_PI/180), sin(p*M_PI/180)) ;
	M3d_mat_mult(mat,  tmp,mat) ;

        M3d_make_z_rotation_cs (tmp, cos(-p*M_PI/180), sin(-p*M_PI/180)) ;
	M3d_mat_mult(inv,  inv,tmp) ;				
        break ;



	
   case TX :
        M3d_make_translation (tmp,   p, 0.0, 0.0 ) ;
	M3d_mat_mult(mat,  tmp,mat) ;

        M3d_make_translation (tmp,  -p, 0.0, 0.0 ) ;
	M3d_mat_mult(inv,  inv,tmp) ;					
        break ;

   case TY :
        M3d_make_translation (tmp,   0.0, p, 0.0) ;
	M3d_mat_mult(mat,  tmp,mat) ;

        M3d_make_translation (tmp,  0.0, -p, 0.0) ;
	M3d_mat_mult(inv,  inv,tmp) ;						
        break ;

   case TZ :
        M3d_make_translation (tmp,   0.0, 0.0, p) ;
	M3d_mat_mult(mat,  tmp,mat) ;

        M3d_make_translation (tmp,  0.0, 0.0, -p) ;
	M3d_mat_mult(inv,  inv,tmp) ; 
        break ;



	
	
   case NX :
        M3d_make_scaling (tmp,  -1.0, 1.0, 1.0 ) ;
	M3d_mat_mult(mat,  tmp,mat) ;

	M3d_mat_mult(inv,  inv,tmp) ; 	
        break ;

   case NY :
        M3d_make_scaling (tmp,   1.0, -1.0, 1.0 ) ;
	M3d_mat_mult(mat,  tmp,mat) ;

	M3d_mat_mult(inv,  inv,tmp) ; 		
        break ;

   case NZ :
        M3d_make_scaling (tmp,   1.0, 1.0, -1.0 ) ;
	M3d_mat_mult(mat,  tmp,mat) ;

	M3d_mat_mult(inv,  inv,tmp) ; 		
        break ;

   default :
        printf("ERROR:  M3d_make_movement_sequence_matrix : unrecognized matrix type = %d\n\n", m) ;
        
        return 0 ;

        break ;   

   }

 }


 return 1 ;

}





//////////////////////////////////////////////////////////////////






int M3d_viewAA (double view[4][4], double view_inverse[4][4],
                double eye[3], double coi[3], double up[3],
                double a, double b, double c, double p, double r)
// return 1 if successful, 0 otherwise
{
    double Uplen ;
    double Up[3] ;

    double A[4][4],B[4][4],C[4][4],D[4][4] ;
    double Ai[4][4],Bi[4][4],Ci[4][4],Di[4][4] ;    
    
    if ((p == 0) || (r == 0)) return 0 ;
    
    M3d_make_translation   (A, -eye[0], -eye[1], -eye[2]) ;
    M3d_make_y_rotation_cs (B, c/p, -a/p) ;
    M3d_make_x_rotation_cs (C, p/r,  b/r) ;
    
    M3d_make_identity(view) ;     
    M3d_mat_mult(view,  A,view) ;
    M3d_mat_mult(view,  B,view) ;    
    M3d_mat_mult(view,  C,view) ;    
    
    M3d_mat_mult_pt (Up, view, up) ;
    Uplen = sqrt(Up[0]*Up[0] + Up[1]*Up[1]) ;
    if (Uplen == 0) return 0 ;
    
    M3d_make_z_rotation_cs(D, Up[1]/Uplen, Up[0]/Uplen) ;
    M3d_mat_mult(view,  D,view) ;    

    // now make the inverse

    M3d_make_translation   (Ai,  eye[0],  eye[1],  eye[2]) ;
    M3d_make_y_rotation_cs (Bi, c/p,  a/p) ;
    M3d_make_x_rotation_cs (Ci, p/r, -b/r) ;
    M3d_make_z_rotation_cs (Di, Up[1]/Uplen, -Up[0]/Uplen) ;
    
    M3d_make_identity(view_inverse) ;     
    M3d_mat_mult(view_inverse,  Di, view_inverse) ;
    M3d_mat_mult(view_inverse,  Ci, view_inverse) ;    
    M3d_mat_mult(view_inverse,  Bi, view_inverse) ;    
    M3d_mat_mult(view_inverse,  Ai, view_inverse) ;        
    
    return 1 ;
}








int M3d_viewBB (double view[4][4], double view_inverse[4][4],
                double eye[3], double coi[3], double up[3],
                double a, double b, double c, double p, double r)
// return 1 if successful, 0 otherwise
{
    double Uplen ;
    double Up[3] ;

    double A[4][4],B[4][4],C[4][4],D[4][4] ;
    double Ai[4][4],Bi[4][4],Ci[4][4],Di[4][4] ;
    
    p = sqrt(b*b + c*c) ; // alter the incoming p

    if ((p == 0) || (r == 0)) return 0 ;

    M3d_make_translation   (A, -eye[0], -eye[1], -eye[2]) ;
    M3d_make_x_rotation_cs (B, c/p,  b/p) ;
    M3d_make_y_rotation_cs (C, p/r, -a/r) ;

    M3d_make_identity(view) ;     
    M3d_mat_mult(view,  A,view) ;
    M3d_mat_mult(view,  B,view) ;    
    M3d_mat_mult(view,  C,view) ;    
    
    M3d_mat_mult_pt (Up, view, up) ;
    Uplen = sqrt(Up[0]*Up[0] + Up[1]*Up[1]) ;
    if (Uplen == 0) return 0 ;
    
    M3d_make_z_rotation_cs (D, Up[1]/Uplen, Up[0]/Uplen) ;
    M3d_mat_mult(view,  D,view) ;    

    // now make the inverse

    M3d_make_translation   (Ai,  eye[0],  eye[1],  eye[2]) ;
    M3d_make_x_rotation_cs (Bi, c/p, -b/p) ;
    M3d_make_y_rotation_cs (Ci, p/r,  a/r) ;
    M3d_make_z_rotation_cs (Di, Up[1]/Uplen, -Up[0]/Uplen) ;
    
    M3d_make_identity(view_inverse) ;     
    M3d_mat_mult(view_inverse,  Di, view_inverse) ;
    M3d_mat_mult(view_inverse,  Ci, view_inverse) ;    
    M3d_mat_mult(view_inverse,  Bi, view_inverse) ;    
    M3d_mat_mult(view_inverse,  Ai, view_inverse) ;
    
    return 1 ;
}
        








int M3d_view (double view[4][4], double view_inverse[4][4],
              double eye[3], double coi[3], double up[3])
// Construct the view matrix and its inverse given the location
// of the eye, the center of interest, and an up point.
// return 1 if successful, 0 otherwise.
{
    double a,b,c,p,r ;
    int s ;


    // printf("entering M3d_view\n") ;
    a = coi[0] - eye[0] ;
    b = coi[1] - eye[1] ;
    c = coi[2] - eye[2] ;
    p = sqrt(a*a + c*c) ;
    r = sqrt(a*a + b*b + c*c) ;

    // printf("a,b,c,p,r = %lf %lf %lf %lf %lf\n\n",a,b,c,p,r) ;

    if (fabs(b) < p) {
      //         printf("choose AA\n") ;
         s = M3d_viewAA (view, view_inverse,
                         eye, coi, up,
                         a,b,c,p,r) ;
    } else {
      //         printf("choose BB\n") ;
         s = M3d_viewBB (view, view_inverse,
                         eye, coi, up,
                         a,b,c,p,r) ;
    }

    if (s == 0) printf("M3d_view error\n") ;

    // printf("leaving M3d_view\n") ;

    return s ;
}







