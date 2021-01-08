#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <libgen.h> //Needed for dirname

#include "M3d_matrix_tools.h"
#include "obj_reader.h"

#include <SDL.h>
#include <SDL_image.h>

char *dir; //Directory path of the current obj file

int num_tris, num_mats, num_vn, num_v;
int current_mat = -1;

double x[100000], y[100000], z[100000], xnormal[50000], ynormal[50000], znormal[50000];
double u[50000], v[50000], w[50000];
Triangle tris[75000];
Material mats[100];

double x_world[100000], y_world[100000], z_world[100000];
double xnormal_world[50000], ynormal_world[50000], znormal_world[50000];

int get_face_format(FILE *f){
  //Finds the format and info provided for the current face

	int a,b,c, ret;
	char d;
	long pos = ftell(f);
	ret = fscanf(f, "%d/%d/%d", &a,&b,&c);
  //Both face of format v and v//vn returns 1
  //Face of format v/vt returns 2
  //Face of format v/vt/vn returns 3
	if(ret == 1){
		fscanf(f, "%c", &d);
		if(d == '/'){
      //Format v//vn
			ret = 4;
		}
    //Else format is v
	}
	fseek(f, pos, SEEK_SET);
	return ret;
}

void find_and_add_normal(int A, int B, int C){
	//Calculates the normal for the face defined by vertices at index A, B and C
	//Adds the normal to the triangle that is currently being read

	double AB[3], AC[3], n[3];

	//Find two vectors defining the current face
	AB[0] = x[B] - x[A];
   	AB[1] = y[B] - y[A];
   	AB[2] = z[B] - z[A];

	AC[0] = x[C] - x[A];
	AC[1] = y[C] - y[A];
	AC[2] = z[C] - z[A];

	M3d_x_product(n, AB, AC);

	//Save normal to global array
	num_vn++;
	xnormal[num_vn] = n[0];
	ynormal[num_vn] = n[1];
	znormal[num_vn] = n[2];

	//Save index of normal to the triangle
	tris[num_tris].An = tris[num_tris].Bn = tris[num_tris].Cn = num_vn;
}

void center_and_scale_object()
{
  // center the object at the origin
  // scale the object to fit inside of a 7x7x7 bounding box
  int index_A, index_B, index_C;
  double xc,yc,zc ;
  double x_min, y_min, z_min, x_max, y_max, z_max;

  xc = yc = zc = 0 ;
  x_min = x_max = x[1];
  y_min = y_max = y[1];
  z_min = z_max = z[1];

  int i ;
  for (i = 1 ; i <= num_v; i++) {
    xc += x[i]; yc += y[i]; zc += z[i];
    
    if(x[i] < x_min) x_min = x[i];
    if(y[i] < y_min) y_min = y[i];
    if(z[i] < z_min) z_min = z[i];

    if(x[i] > x_max) x_max = x[i];
    if(y[i] > y_max) y_max = y[i];
    if(z[i] > z_max) z_max = z[i];
  }
  xc /= num_v; yc /= num_v; zc /= num_v;

  double xd, yd, zd;

  //Find the biggest distance between two points in all three axis.
  xd = x_max - x_min;
  yd = y_max - y_min;
  zd = z_max - z_min;

  double sf, delta;

  if((xd > yd) && (xd > zd)) delta = xd;
  else if(yd > zd) delta = yd;
  else delta = zd;

  //Set scaling factor based on largest distance two fit all vertices in a 7x7x7 cube.
  sf = 7/delta;

  for (i = 1 ; i <= num_v ; i++) {
    x[i] -= xc;  y[i] -= yc;  z[i] -= zc;
    x[i] *= sf;  y[i] *= sf;  z[i] *= sf;
  }
}

void init_mat(Material mat)
{
	//Default colors of materials
	mat.Ka[0] = 1.0; mat.Ka[1] = 1.0; mat.Ka[2] = 1.0;
	mat.Kd[0] = 1.0; mat.Kd[1] = 1.0; mat.Kd[2] = 1.0;
	mat.Ks[0] = 1.0; mat.Ks[1] = 1.0; mat.Ks[2] = 1.0;

	mat.map_Ka = NULL;
	mat.map_Kd = NULL;
	mat.map_Ks = NULL;
}

int read_mtl_file(char *fname){
	
  //Parses .mtl material file
	FILE *f;
	SDL_Surface *image;
	char keyword[100];
	char name[100];
	char path[100];
	double dump;
	f = fopen(fname, "rb");
	if(f == NULL){
		SDL_Log("can't open file, %s\n", fname);
	}
	while(fscanf(f, "%s", keyword) == 1){
		if(strcmp(keyword, "newmtl") == 0){
			fscanf(f, "%s", mats[num_mats].name);
			init_mat(mats[num_mats]);
			num_mats++;
		}
    else if(strcmp(keyword, "Ka") == 0){//ambient rgb
    	fscanf(f, "%lf %lf %lf", &mats[num_mats-1].Ka[0], &mats[num_mats-1].Ka[1], &mats[num_mats-1].Ka[2]);
    }
    else if(strcmp(keyword, "Kd") == 0){//diffuse rgb
    	fscanf(f, "%lf %lf %lf", &mats[num_mats-1].Kd[0], &mats[num_mats-1].Kd[1], &mats[num_mats-1].Kd[2]);
    }
    else if(strcmp(keyword, "Ks") == 0){//specular rgb
    	fscanf(f, "%lf %lf %lf", &mats[num_mats-1].Ks[0], &mats[num_mats-1].Ks[1], &mats[num_mats-1].Ks[2]);
    }
    else if(strcmp(keyword, "Ke") == 0){//specular rgb
    	fscanf(f, "%lf %lf %lf", &mats[num_mats-1].Ke[0], &mats[num_mats-1].Ke[1], &mats[num_mats-1].Ke[2]);
    }
    else if(strcmp(keyword, "map_Ka") == 0){//ambient texture map
    	fscanf(f, "%s", name);
    	strcpy(path, dir);
    	strcat(path, name);
    	image = IMG_Load(path);
    	if(!image)
    	{
    		SDL_Log("IMG_Load: %s\n", IMG_GetError());
    	}
    	else
    	{
    		mats[num_mats-1].map_Ka = image;
    	}
    }
    else if(strcmp(keyword, "map_Kd") == 0){//diffuse texture map
    	fscanf(f, "%s", name);
    	strcpy(path, dir);
    	strcat(path, name);
    	image = IMG_Load(path);
    	if(!image)
    	{
    		SDL_Log("IMG_Load: %s\n", IMG_GetError());
    	}
    	else
    	{
    		mats[num_mats-1].map_Kd = image;
    	}
    }
    else if(strcmp(keyword, "map_Ks") == 0){//specular texture map
    	fscanf(f, "%s", name);
    	strcpy(path, dir);
    	strcat(path, name);
    	image = IMG_Load(path);
    	if(!image)
    	{
    		SDL_Log("IMG_Load: %s\n", IMG_GetError());
    	}
    	else
    	{
    		mats[num_mats-1].map_Ks = image;
    	}
    }
    else if(strcmp(keyword, "map_bump") == 0){//specular texture map
    	fgets(name, 100, f);
    }
    else if(strcmp(keyword, "Tr") == 0){
    	fscanf(f, "%lf", &dump);
    }
    else if(strcmp(keyword, "Tf") == 0){
    	fscanf(f, "%lf %lf %lf", &mats[num_mats-1].Tf[0], &mats[num_mats-1].Tf[1], &mats[num_mats-1].Tf[2]);
    }
    else if(strcmp(keyword, "illum") == 0){
    	fscanf(f, "%d", &mats[num_mats-1].illum);
    }
    else if(strcmp(keyword, "Ns") == 0){
    	fscanf(f, "%lf", &mats[num_mats-1].Ns);
    }
    else if(strcmp(keyword, "Ni") == 0){
    	fscanf(f, "%lf" , &mats[num_mats-1].Ni);
    }
    else if(strcmp(keyword, "d") == 0){
    	fscanf(f,  "%lf", &mats[num_mats-1].d);
    }
    else if(strcmp(keyword, "#") == 0){//comment
    	fgets(name, 100, f);
    }
    else{
    	fgets(name, 100, f);
    }
}
return 1;
}

int read_obj_file(const char *fname)
{
  //Parses .obj object file
	num_tris = 0; num_mats = 0; num_vn = 0; num_v = 0;
	int num_vt = 0; 
	int index_vA, index_vB, index_vC, index_vD, index_vt, index_vn, i;
	int format;
	double n[3], AB[3], AC[3];
	FILE *f;
	char keyword[100];
	char name[100];
	char path[100];

	//Make duplicate of fname to be used to find directory name
	char *dup = strdup(fname);
	char ch = '/';

	SDL_Log("Loading obj file %s\n", fname);

	dir = dirname(dup);
	strcat(dir, &ch);

	//Need to read as binary for ftell and fseek to work properly on Windows
	f = fopen(fname, "rb");
	if(f == NULL)
	{
		SDL_Log("can't open file, %s\n", fname);
		exit(1);
	}
	num_tris = 0;
	while(fscanf(f, "%s", keyword) == 1)
	{
		if(strcmp(keyword, "v") == 0)
    {//geometric vertex
    	num_v++;
    	fscanf(f, "%lf %lf %lf %lf", &x[num_v], &y[num_v], &z[num_v], &w[num_v]);
    }
    else if(strcmp(keyword, "vt") == 0)
    {//texture vertex
    	num_vt++;
    	fscanf(f, "%lf %lf %lf", &u[num_vt], &v[num_vt], &w[num_vt]);
    }
    else if(strcmp(keyword, "vn") == 0)
    {//vertex normal
    	num_vn++;
    	fscanf(f, "%lf %lf %lf", &xnormal[num_vn], &ynormal[num_vn], &znormal[num_vn]);
    }
    else if(strcmp(keyword, "vp") == 0)
    {//parameter space vertex
    }
    else if(strcmp(keyword, "g") == 0)
    {//group
    	fscanf(f, "%s", name);
    }
    else if(strcmp(keyword, "s") == 0)
    {//smooth shading
    	fscanf(f, "%s", name);
    }
    else if(strcmp(keyword, "f") == 0)
    {//face
    	tris[num_tris].mtl.index = current_mat;
    	if(tris[num_tris].mtl.index != -1)
    	{
    		tris[num_tris].mtl = mats[current_mat];
    	}
      //Checking what information is included for the face
    	format = get_face_format(f);

    	if(format == 1)
    	{//Face format: v v v
			//Indeces of vertices of the triangle
    		fscanf(f, "%d", &index_vA);
    		fscanf(f, "%d", &index_vB);
    		fscanf(f, "%d", &index_vC);

			tris[num_tris].A = index_vA;
			tris[num_tris].B = index_vB;
			tris[num_tris].C = index_vC;

			//Normal vectors not provided
			//Calculated here, same normal added to all three vertices
    		find_and_add_normal(index_vA, index_vB, index_vC);

			//Check if face is a quadrilateral
    		if(fscanf(f, "%d", &index_vD) == 1){
	  			//Triangulate the quadrilateral
    			num_tris++;

	  			//Indeces of vertices of second triangle
    			tris[num_tris].A = index_vA;
    			tris[num_tris].B = index_vC;
    			tris[num_tris].C = index_vD;

	  			//Normal vector for the vertices not provided by file
	  			//Normal vector for surface is the same as 1st triangle
    			tris[num_tris].An = tris[num_tris].Bn = tris[num_tris].Cn = num_vn;

    		}
    	}
      else if(format == 2){//Face format: v/vt v/vt v/vt
		//First vertex of triangle
      	fscanf(f, "%d/%d", &index_vA, &index_vt);

      	tris[num_tris].A = index_vA;
      	tris[num_tris].At = index_vt;

		//Second vertex of triangle
      	fscanf(f, "%d/%d", &index_vB, &index_vt);

      	tris[num_tris].B = index_vB;
      	tris[num_tris].Bt= index_vt;

		//Third vertex of triangle
      	fscanf(f, "%d/%d", &index_vC, &index_vt);

      	tris[num_tris].C = index_vC;
      	tris[num_tris].Ct = index_vt;

		//Normal vectors not provided
		//Calculated here, same normal added to all three vertices
      	find_and_add_normal(index_vA, index_vB, index_vC);

		//Check if face is a quadrilateral
      	if(fscanf(f, "%d/%d", &index_vD, &index_vt) == 2){
	  		//Triangulate the quadrilateral
      		num_tris++;

	  		//First vertex of second triangle
      		tris[num_tris].A = index_vA;

	  		//Shares texture vertices with first vertex of 1st triangle
      		tris[num_tris].At = tris[num_tris-1].At;

	  		//Second vertex of second triangle
      		tris[num_tris].B = index_vC;

	  		//Shares texture vertices with third vertex of 1st triangle
      		tris[num_tris].Bt = tris[num_tris-1].Ct;

	  		//Third vertex of second triangle
      		tris[num_tris].C = index_vD;
      		tris[num_tris].Ct = index_vt;

	  		//Normals vector for the vertices not provided by file
	  		//Normal vector for surface is the same as 1st triangle
      		tris[num_tris].An = tris[num_tris].Bn = tris[num_tris].Cn = num_vn;
      	}
      }
      else if(format == 3){//Face format: v/vt/vn  v/vt/vn v/vt/vn
		//First vertex of triangle
      	fscanf(f, "%d/%d/%d", &index_vA, &index_vt, &index_vn);

      	tris[num_tris].A = index_vA;
      	tris[num_tris].At = index_vt;
      	tris[num_tris].An = index_vn;

		//Second vertex of triangle
      	fscanf(f, "%d/%d/%d", &index_vB, &index_vt, &index_vn);

      	tris[num_tris].B = index_vB;
      	tris[num_tris].Bt = index_vt;
      	tris[num_tris].Bn = index_vn;

		//Third vertex of triangle
      	fscanf(f, "%d/%d/%d", &index_vC, &index_vt, &index_vn);

      	tris[num_tris].C = index_vC;
      	tris[num_tris].Ct = index_vt;
      	tris[num_tris].Cn = index_vn;

		//Check if face is a quadrilateral
      	if(fscanf(f, "%d/%d/%d", &index_vD, &index_vt, &index_vn) == 3){
	  		//Triangulate the quadrilateral
      		num_tris++;

	  		//First vertex of second triangle
      		tris[num_tris].A = index_vA;

	  		//Shares texture vertices and normal vector with first vertex of 1st triangle
      		tris[num_tris].At = tris[num_tris-1].At;
      		tris[num_tris].An = tris[num_tris-1].An;

	  		//Second vertex of second triangle
      		tris[num_tris].B = index_vC;

	  		//Shares texture vertices and normal vector with third vertex of 1st triangle
      		tris[num_tris].Bt = tris[num_tris-1].Ct;
      		tris[num_tris].Bn = tris[num_tris-1].Cn;

	  		//Third vertex of second triangle
      		tris[num_tris].C = index_vD;
      		tris[num_tris].Ct = index_vt;
      		tris[num_tris].Cn = index_vn;
      	}
      }
      else if(format == 4){//Face format: v//vn v//vn v//vn
		//First vertex of triangle
      	fscanf(f, "%d//%d", &index_vA, &index_vn);

      	tris[num_tris].A = index_vA;
      	tris[num_tris].An = index_vn;

		//Second vertex of triangle
      	fscanf(f, "%d//%d", &index_vB, &index_vn);

      	tris[num_tris].B = index_vB;
      	tris[num_tris].Bn = index_vn;

		//Third vertex of triangle
      	fscanf(f, "%d//%d", &index_vC, &index_vn);

      	tris[num_tris].C = index_vC;
      	tris[num_tris].Cn = index_vn;

		//Check if face is a quadrilateral
      	if(fscanf(f, "%d//%d", &index_vD, &index_vn) == 2){
	  		//Triangulate the quadrilateral
      		num_tris++;

	  		//First vertex of second triangle
      		tris[num_tris].A = index_vA;
      		tris[num_tris].An = tris[num_tris-1].An;

	  		//Second vertex of second triangle
      		tris[num_tris].B = index_vC;
      		tris[num_tris].Bn = tris[num_tris-1].Cn;


	  		//Third vertex of second triangle
      		tris[num_tris].C = index_vD;
      		tris[num_tris].Cn = index_vn;

      	}
      }
      tris[num_tris].mtl.index = current_mat;
      if(tris[num_tris].mtl.index != -1){
      	tris[num_tris].mtl = mats[current_mat];
      }
      
      num_tris++;
  }
    else if(strcmp(keyword, "mtllib") == 0){//Material file
    	fscanf(f, "%s", name);
    	strcpy(path, dir);

    	strcat(path, name);
    	read_mtl_file(path);
    }
    else if(strcmp(keyword, "usemtl") == 0){//Use material
    	fscanf(f, "%s", name);
    	for(i = 0; i < num_mats; i++){
    		if(strcmp(name, mats[i].name) == 0){
    			current_mat = i;
    			break;
    		}
    	}
    }
    else if(strcmp(keyword, "#") == 0){//comment
      fgets(name, 100, f); //skip the line
  }
}
center_and_scale_object();
SDL_Log("Triangles: %d\n", num_tris);
SDL_Log("Finished loading %s\n", fname);
return 1;
}
