#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "M3d_matrix_tools.h"
#include "obj_reader.h"
#include "kernel.h"

#include <SDL.h>
#include <SDL_image.h>

#define MAX_KEYWORD_LENGTH 100

char drive[_MAX_DRIVE]; //Drive of current obj file
char dir[_MAX_DIR]; //Directory of the current obj file

int num_tris, num_mats, num_vn, num_v;
int current_mat = -1;

double* x, * y, * z, * xnormal, * ynormal, * znormal;
double* u, * v;
Triangle* tris;
Material* mats;

int get_face_format(FILE* f) {
	//Finds the format and info provided for the current face
	//Returns 1 for v, 2 for v/vt, 3 for v/vt/vn and 4 for v//vn

	int a, b, c, ret;
	char d;
	long pos = ftell(f);
	ret = fscanf_s(f, "%d/%d/%d", &a, &b, &c);
	//Both face of format v and v//vn returns 1
	//Face of format v/vt returns 2
	//Face of format v/vt/vn returns 3
	if (ret == 1) {
		fscanf_s(f, "%c", &d, 1);
		if (d == '/') {
			//Format v//vn
			ret = 4;
		}
		//Else format is v
	}
	fseek(f, pos, SEEK_SET);
	return ret;
}

void find_and_add_normal(int A, int B, int C) {
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
	// scale the object to fit inside of a 15x15x15 bounding box
	double xc, yc, zc;
	double x_min, y_min, z_min, x_max, y_max, z_max;

	xc = yc = zc = 0;
	x_min = x_max = x[1];
	y_min = y_max = y[1];
	z_min = z_max = z[1];

	int i;

	for (i = 1; i <= num_v; i++) {
		xc += x[i]; yc += y[i]; zc += z[i];

		if (x[i] < x_min) x_min = x[i];
		if (y[i] < y_min) y_min = y[i];
		if (z[i] < z_min) z_min = z[i];

		if (x[i] > x_max) x_max = x[i];
		if (y[i] > y_max) y_max = y[i];
		if (z[i] > z_max) z_max = z[i];
	}
	xc /= num_v; yc /= num_v; zc /= num_v;

	double xd, yd, zd;

	//Find the biggest distance between two points in all three axis.
	xd = x_max - x_min;
	yd = y_max - y_min;
	zd = z_max - z_min;

	double sf, delta;

	if ((xd > yd) && (xd > zd)) delta = xd;
	else if (yd > zd) delta = yd;
	else delta = zd;

	//Set scaling factor based on largest distance to fit all vertices in a 15x15x15 cube.
	sf = 15 / delta;

	double t[4][4], s[4][4], m[4][4];

	M3d_make_translation(t, -xc, -yc, -zc);
	M3d_make_scaling(s, sf, sf, sf);
	M3d_mat_mult(m, s, t);

	mat_mult_device(x, y, z, m, x, y, z, num_v + 1);
}

void init_mat()
{
	mats[num_mats].index = num_mats;

	//Default colors of materials
	mats[num_mats].Ka[0] = 1.0; mats[num_mats].Ka[1] = 1.0; mats[num_mats].Ka[2] = 1.0;
	mats[num_mats].Kd[0] = 1.0; mats[num_mats].Kd[1] = 1.0; mats[num_mats].Kd[2] = 1.0;
	mats[num_mats].Ks[0] = 1.0; mats[num_mats].Ks[1] = 1.0; mats[num_mats].Ks[2] = 1.0;

	mats[num_mats].map_Ka = NULL;
	mats[num_mats].map_Kd = NULL;
	mats[num_mats].map_Ks = NULL;
}

//Returns the number of new materials in .mtl file at fname
int allocate_mtl_mem(char* fname) {
	FILE* f;
	errno_t err;
	char keyword[MAX_KEYWORD_LENGTH];

	int num = 0;

	err = fopen_s(&f, fname, "rb");
	if (err != 0) {
		SDL_Log("can't open file, %s\n", fname);
		exit(1);
	}
	while (fscanf_s(f, "%s", keyword, MAX_KEYWORD_LENGTH) == 1) {
		if (strcmp(keyword, "newmtl") == 0) {
			num++;
		}
		else {
			//Don't need to allocate memory for anything else, 
			//skip line
			fgets(keyword, MAX_KEYWORD_LENGTH, f);
		}
	}
	return num;
}

int read_mtl_file(char* fname) {
	//Parses .mtl material file
	FILE* f;
	errno_t err;
	SDL_Surface* image;
	char keyword[MAX_KEYWORD_LENGTH];
	char name[_MAX_FNAME];
	char path[_MAX_PATH];
	double dump;
	err = fopen_s(&f, fname, "rb");
	if (err != 0) {
		SDL_Log("can't open file, %s\n", fname);
		exit(1);
	}
	while (fscanf_s(f, "%s", keyword, MAX_KEYWORD_LENGTH) == 1) {
		if (strcmp(keyword, "newmtl") == 0) {
			fscanf_s(f, "%s", mats[num_mats].name, _MAX_FNAME);
			init_mat();
			num_mats++;
		}
		else if (strcmp(keyword, "Ka") == 0) {//ambient rgb
			fscanf_s(f, "%lf %lf %lf", &mats[num_mats - 1].Ka[0], &mats[num_mats - 1].Ka[1], &mats[num_mats - 1].Ka[2]);
		}
		else if (strcmp(keyword, "Kd") == 0) {//diffuse rgb
			fscanf_s(f, "%lf %lf %lf", &mats[num_mats - 1].Kd[0], &mats[num_mats - 1].Kd[1], &mats[num_mats - 1].Kd[2]);
		}
		else if (strcmp(keyword, "Ks") == 0) {//specular rgb
			fscanf_s(f, "%lf %lf %lf", &mats[num_mats - 1].Ks[0], &mats[num_mats - 1].Ks[1], &mats[num_mats - 1].Ks[2]);
		}
		else if (strcmp(keyword, "Ke") == 0) {//specular rgb
			fscanf_s(f, "%lf %lf %lf", &mats[num_mats - 1].Ke[0], &mats[num_mats - 1].Ke[1], &mats[num_mats - 1].Ke[2]);
		}
		else if (strcmp(keyword, "map_Ka") == 0) {//ambient texture map
			fscanf_s(f, "%s", name, _MAX_FNAME);
			strcpy_s(path, _MAX_PATH, dir);
			strcat_s(path, _MAX_PATH, name);
			image = IMG_Load(path);
			if (!image)
			{
				SDL_Log("IMG_Load: %s\n", IMG_GetError());
			}
			else
			{
				mats[num_mats - 1].map_Ka = image;
			}
		}
		else if (strcmp(keyword, "map_Kd") == 0) {//diffuse texture map
			fscanf_s(f, "%s", name, _MAX_FNAME);
			strcpy_s(path, _MAX_PATH, dir);
			strcat_s(path, _MAX_PATH, name);
			image = IMG_Load(path);
			if (!image)
			{
				SDL_Log("IMG_Load: %s\n", IMG_GetError());
			}
			else
			{
				mats[num_mats - 1].map_Kd = image;
			}
		}
		else if (strcmp(keyword, "map_Ks") == 0) {//specular texture map
			strcpy_s(path, _MAX_PATH, dir);
			strcat_s(path, _MAX_PATH, name);
			image = IMG_Load(path);
			if (!image)
			{
				SDL_Log("IMG_Load: %s\n", IMG_GetError());
			}
			else
			{
				mats[num_mats - 1].map_Ks = image;
			}
		}
		else if (strcmp(keyword, "map_bump") == 0) {
			fgets(name, 100, f);
		}
		else if (strcmp(keyword, "Tr") == 0) {
			fscanf_s(f, "%lf", &dump);
		}
		else if (strcmp(keyword, "Tf") == 0) {
			fscanf_s(f, "%lf %lf %lf", &mats[num_mats - 1].Tf[0], &mats[num_mats - 1].Tf[1], &mats[num_mats - 1].Tf[2]);
		}
		else if (strcmp(keyword, "illum") == 0) {
			fscanf_s(f, "%d", &mats[num_mats - 1].illum);
		}
		else if (strcmp(keyword, "Ns") == 0) {
			fscanf_s(f, "%lf", &mats[num_mats - 1].Ns);
		}
		else if (strcmp(keyword, "Ni") == 0) {
			fscanf_s(f, "%lf", &mats[num_mats - 1].Ni);
		}
		else if (strcmp(keyword, "d") == 0) {
			fscanf_s(f, "%lf", &mats[num_mats - 1].d);
		}
		else if (strcmp(keyword, "#") == 0) {
			fgets(name, 100, f);
		}
		else {
			fgets(name, 100, f);
		}
	}
	return 1;
}

//Scans through the .obj file and allocates the necessary memory
//Exits if it reads faces with more than 4 vertices
void allocate_mem(const char* fname) {
	num_tris = num_mats = num_vn = 0;
	int num_vt = 0;
	int format;
	int a[3];

	FILE* f;
	errno_t err;
	char keyword[MAX_KEYWORD_LENGTH], path[_MAX_PATH], name[_MAX_FNAME];

	err = fopen_s(&f, fname, "rb");
	if (err != 0)
	{
		SDL_Log("can't open file, %s\n", fname);
		exit(1);
	}

	while (fscanf_s(f, "%s", keyword, MAX_KEYWORD_LENGTH) == 1) {
		if (strcmp(keyword, "v") == 0) {
			num_v++;
		}
		else if (strcmp(keyword, "vt") == 0) {
			num_vt++;
		}
		else if (strcmp(keyword, "vn") == 0) {
			num_vn++;
		}
		else if (strcmp(keyword, "f") == 0) {
			format = get_face_format(f);
			if (format == 1) {//Face format: v v v
				//A face must have at least three vertices, so skip past these and increment triangle count
				fscanf_s(f, "%d %d %d", &a[0], &a[1], &a[2]);
				num_tris++;

				//Will need to calculate and add 1 normal for the face
				num_vn++;

				if (fscanf_s(f, "%d", &a[0]) == 1) {
					//Face has a fourth vertex, means a quadrilateral
					//Will be triangulated into two triangles
					num_tris++;


					if (fscanf_s(f, "%d", &a[0]) == 1) {
						//Face has a fifth vertex
						//Program can't currently handle faces with more than 4 vertices
						SDL_Log("The object contains a face with more than 4 vertices. Exiting\n");
						exit(1);
					}
				}
			}
			else if (format == 2) {//Face format: v/vt v/vt v/vt
				//Skip past the first three vertices
				fscanf_s(f, "%d/%d", &a[0], &a[1]);
				fscanf_s(f, "%d/%d", &a[0], &a[1]);
				fscanf_s(f, "%d/%d", &a[0], &a[1]);
				num_tris++;

				//Will need to calculate and add 1 normal for the face
				num_vn++;

				if (fscanf_s(f, "%d/%d", &a[0], &a[1]) == 2) {
					//Face has a fourth vertex, means a quadrilateral
					num_tris++;
					if (fscanf_s(f, "%d/%d", &a[0], &a[1]) == 2) {
						//Fifth vertex
						SDL_Log("The object contains a face with more than 4 vertices. Exiting\n");
						exit(1);
					}
				}
			}
			else if (format == 3) {//Face format: v/vt/vn  v/vt/vn v/vt/vn
				//Skip past the first three vertices
				fscanf_s(f, "%d/%d/%d", &a[0], &a[1], &a[2]);
				fscanf_s(f, "%d/%d/%d", &a[0], &a[1], &a[2]);
				fscanf_s(f, "%d/%d/%d", &a[0], &a[1], &a[2]);
				num_tris++;

				if (fscanf_s(f, "%d/%d/%d", &a[0], &a[1], &a[2]) == 3) {
					//Face has a fourth vertex, means a quadrilateral
					num_tris++;
					if (fscanf_s(f, "%d/%d/%d", &a[0], &a[1], &a[2]) == 3) {
						//Fifth vertex
						SDL_Log("The object contains a face with more than 4 vertices. Exiting\n");
						exit(1);
					}
				}
			}
			else if (format == 4) {//Face format: v//vn v//vn v//vn
				//Skip past the first three vertices
				fscanf_s(f, "%d//%d", &a[0], &a[1]);
				fscanf_s(f, "%d//%d", &a[0], &a[1]);
				fscanf_s(f, "%d//%d", &a[0], &a[1]);
				num_tris++;

				if (fscanf_s(f, "%d//%d", &a[0], &a[1]) == 2) {
					//Face has a fourth vertex, means a quadrilateral
					num_tris++;
					if (fscanf_s(f, "%d//%d", &a[0], &a[1]) == 2) {
						//Fifth vertex
						SDL_Log("The object contains a face with more than 4 vertices. Exiting\n");
						exit(1);
					}
				}
			}
		}
		else if (strcmp(keyword, "mtllib") == 0) {
			fscanf_s(f, "%s", name, _MAX_FNAME);

			errno_t err = _makepath_s(path, _MAX_PATH, drive, dir, name, NULL);
			if (err != 0) {
				printf("Error creating path. Error code %d.\n", err);
				exit(1);
			}

			num_mats += allocate_mtl_mem(path);
		}
		else {
			//Read something that is not currently being allocated memory
			fgets(keyword, MAX_KEYWORD_LENGTH, f); //skip the line
		}
	}
	//.obj is 1-indexed so need to be 1 larget than the number of vertices
	num_v++; num_vn++; num_vt++;

	//Vertices in object space
	x = (double*)malloc(num_v * sizeof(double));
	y = (double*)malloc(num_v * sizeof(double));
	z = (double*)malloc(num_v * sizeof(double));

	if ((x == NULL) || (y == NULL) || (z == NULL)) {
		SDL_Log("Unable to allocate memory for vertices\n");
		exit(1);
	}

	//uv texture coordinates
	u = (double*)malloc(num_vt * sizeof(double));
	v = (double*)malloc(num_vt * sizeof(double));

	if ((u == NULL) || (v == NULL)) {
		SDL_Log("Unable to allocate memory for texure coordinates\n");
		exit(1);
	}

	//Normal vectors in object space
	xnormal = (double*)malloc(num_vn * sizeof(double));
	ynormal = (double*)malloc(num_vn * sizeof(double));
	znormal = (double*)malloc(num_vn * sizeof(double));

	if ((xnormal == NULL) || (ynormal == NULL) || (znormal == NULL)) {
		SDL_Log("Unable to allocate memory for normal vectors\n");
		exit(1);
	}

	tris = (Triangle*)malloc(num_tris * sizeof(Triangle));

	if (tris == NULL) {
		SDL_Log("Unable to allocate memory for triangles\n");
		exit(1);
	}

	mats = (Material*)malloc(num_mats * sizeof(Material));

	if (mats == NULL) {
		SDL_Log("Unable to allocate memory for materials\n");
		exit(1);
	}

}

//Parses .obj object file
int read_obj_file(const char* fname)
{
	//Find drive and directory of .obj file
	//Used to find related materials and resources
	errno_t err = _splitpath_s(fname, drive, _MAX_DRIVE, dir, _MAX_DIR, NULL, 0, NULL, 0);
	if (err != 0) {
		SDL_Log("Error splitting the path. Error code %d.\n", err);
		exit(1);
	}


	allocate_mem(fname);

	num_tris = 0; num_mats = 0; num_vn = 0; num_v = 0;
	int num_vt = 0;
	int index_vA, index_vB, index_vC, index_vD, index_vt, index_vn, i;
	int format;
	double w;
	FILE* f;
	char keyword[MAX_KEYWORD_LENGTH];
	char name[_MAX_FNAME];
	char path[_MAX_PATH];



	SDL_Log("Loading obj file %s\n", fname);


	//Need to read as binary for ftell and fseek to work properly on Windows
	err = fopen_s(&f, fname, "rb");
	if (f == NULL)
	{
		SDL_Log("can't open file, %s\n", fname);
		exit(1);
	}
	num_tris = 0;
	while (fscanf_s(f, "%s", keyword, MAX_KEYWORD_LENGTH) == 1)
	{
		if (strcmp(keyword, "v") == 0)
		{//geometric vertex
			num_v++;
			fscanf_s(f, "%lf %lf %lf %lf", &x[num_v], &y[num_v], &z[num_v], &w);
		}
		else if (strcmp(keyword, "vt") == 0)
		{//texture vertex
			num_vt++;
			fscanf_s(f, "%lf %lf", &u[num_vt], &v[num_vt]);
		}
		else if (strcmp(keyword, "vn") == 0)
		{//vertex normal
			num_vn++;
			fscanf_s(f, "%lf %lf %lf", &xnormal[num_vn], &ynormal[num_vn], &znormal[num_vn]);
		}
		else if (strcmp(keyword, "vp") == 0)
		{//parameter space vertex
		}
		else if (strcmp(keyword, "g") == 0)
		{//group
			fscanf_s(f, "%s", name, _MAX_FNAME);
		}
		else if (strcmp(keyword, "s") == 0)
		{//smooth shading
			fscanf_s(f, "%s", name, _MAX_FNAME);
		}
		else if (strcmp(keyword, "f") == 0)
		{//face
			tris[num_tris].mtl.index = current_mat;
			if (tris[num_tris].mtl.index != -1)
			{
				tris[num_tris].mtl = mats[current_mat];
			}


			//Checking what information is included for the face
			format = get_face_format(f);

			if (format == 1)
			{//Face format: v v v
				//Indeces of vertices of the triangle
				fscanf_s(f, "%d", &index_vA);
				fscanf_s(f, "%d", &index_vB);
				fscanf_s(f, "%d", &index_vC);

				tris[num_tris].A = index_vA;
				tris[num_tris].B = index_vB;
				tris[num_tris].C = index_vC;

				//Normal vectors not provided
				//Calculated here, same normal added to all three vertices
				find_and_add_normal(index_vA, index_vB, index_vC);

				//Check if face is a quadrilateral
				if (fscanf_s(f, "%d", &index_vD) == 1) {
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
			else if (format == 2) {//Face format: v/vt v/vt v/vt
			  //First vertex of triangle
				fscanf_s(f, "%d/%d", &index_vA, &index_vt);

				tris[num_tris].A = index_vA;
				tris[num_tris].At = index_vt;

				//Second vertex of triangle
				fscanf_s(f, "%d/%d", &index_vB, &index_vt);

				tris[num_tris].B = index_vB;
				tris[num_tris].Bt = index_vt;

				//Third vertex of triangle
				fscanf_s(f, "%d/%d", &index_vC, &index_vt);

				tris[num_tris].C = index_vC;
				tris[num_tris].Ct = index_vt;

				//Normal vectors not provided
				//Calculated here, same normal added to all three vertices
				find_and_add_normal(index_vA, index_vB, index_vC);

				//Check if face is a quadrilateral
				if (fscanf_s(f, "%d/%d", &index_vD, &index_vt) == 2) {
					//Triangulate the quadrilateral
					num_tris++;

					//First vertex of second triangle
					tris[num_tris].A = index_vA;

					//Shares texture vertices with first vertex of 1st triangle
					tris[num_tris].At = tris[num_tris - 1].At;

					//Second vertex of second triangle
					tris[num_tris].B = index_vC;

					//Shares texture vertices with third vertex of 1st triangle
					tris[num_tris].Bt = tris[num_tris - 1].Ct;

					//Third vertex of second triangle
					tris[num_tris].C = index_vD;
					tris[num_tris].Ct = index_vt;

					//Normals vector for the vertices not provided by file
					//Normal vector for surface is the same as 1st triangle
					tris[num_tris].An = tris[num_tris].Bn = tris[num_tris].Cn = num_vn;
				}
			}
			else if (format == 3) {//Face format: v/vt/vn  v/vt/vn v/vt/vn
			  //First vertex of triangle
				fscanf_s(f, "%d/%d/%d", &index_vA, &index_vt, &index_vn);

				tris[num_tris].A = index_vA;
				tris[num_tris].At = index_vt;
				tris[num_tris].An = index_vn;

				//Second vertex of triangle
				fscanf_s(f, "%d/%d/%d", &index_vB, &index_vt, &index_vn);

				tris[num_tris].B = index_vB;
				tris[num_tris].Bt = index_vt;
				tris[num_tris].Bn = index_vn;

				//Third vertex of triangle
				fscanf_s(f, "%d/%d/%d", &index_vC, &index_vt, &index_vn);

				tris[num_tris].C = index_vC;
				tris[num_tris].Ct = index_vt;
				tris[num_tris].Cn = index_vn;

				//Check if face is a quadrilateral
				if (fscanf_s(f, "%d/%d/%d", &index_vD, &index_vt, &index_vn) == 3) {
					//Triangulate the quadrilateral
					num_tris++;

					//First vertex of second triangle
					tris[num_tris].A = index_vA;

					//Shares texture vertices and normal vector with first vertex of 1st triangle
					tris[num_tris].At = tris[num_tris - 1].At;
					tris[num_tris].An = tris[num_tris - 1].An;

					//Second vertex of second triangle
					tris[num_tris].B = index_vC;

					//Shares texture vertices and normal vector with third vertex of 1st triangle
					tris[num_tris].Bt = tris[num_tris - 1].Ct;
					tris[num_tris].Bn = tris[num_tris - 1].Cn;

					//Third vertex of second triangle
					tris[num_tris].C = index_vD;
					tris[num_tris].Ct = index_vt;
					tris[num_tris].Cn = index_vn;
				}
			}
			else if (format == 4) {//Face format: v//vn v//vn v//vn
			  //First vertex of triangle
				fscanf_s(f, "%d//%d", &index_vA, &index_vn);

				tris[num_tris].A = index_vA;
				tris[num_tris].An = index_vn;

				//Second vertex of triangle
				fscanf_s(f, "%d//%d", &index_vB, &index_vn);

				tris[num_tris].B = index_vB;
				tris[num_tris].Bn = index_vn;

				//Third vertex of triangle
				fscanf_s(f, "%d//%d", &index_vC, &index_vn);

				tris[num_tris].C = index_vC;
				tris[num_tris].Cn = index_vn;

				//Check if face is a quadrilateral
				if (fscanf_s(f, "%d//%d", &index_vD, &index_vn) == 2) {
					//Triangulate the quadrilateral
					num_tris++;

					//First vertex of second triangle
					tris[num_tris].A = index_vA;
					tris[num_tris].An = tris[num_tris - 1].An;

					//Second vertex of second triangle
					tris[num_tris].B = index_vC;
					tris[num_tris].Bn = tris[num_tris - 1].Cn;


					//Third vertex of second triangle
					tris[num_tris].C = index_vD;
					tris[num_tris].Cn = index_vn;

				}
			}
			tris[num_tris].mtl.index = current_mat;
			if (tris[num_tris].mtl.index != -1) {
				tris[num_tris].mtl = mats[current_mat];
			}

			num_tris++;
		}
		else if (strcmp(keyword, "mtllib") == 0) {//Material file
			fscanf_s(f, "%s", name, _MAX_FNAME);
			err = _makepath_s(path, _MAX_PATH, drive, dir, name, NULL);
			if (err != 0) {
				printf("Error creating path. Error code %d.\n", err);
				exit(1);
			}

			read_mtl_file(path);
		}
		else if (strcmp(keyword, "usemtl") == 0) {//Use material
			fscanf_s(f, "%s", name, _MAX_FNAME);
			for (i = 0; i < num_mats; i++) {
				if (strcmp(name, mats[i].name) == 0) {
					current_mat = i;
					break;
				}
			}
		}
		else if (strcmp(keyword, "#") == 0) {//comment
			fgets(name, 100, f); //skip the line
		}
	}

	center_and_scale_object();
	SDL_Log("Triangles: %d\n", num_tris);
	SDL_Log("Finished loading %s\n", fname);
	return 1;
}


//Frees all memory used by the current object
void close_object() {
	free(x);
	free(y);
	free(z);

	free(xnormal);
	free(ynormal);
	free(znormal);

	free(u);
	free(v);

	free(tris);
	free(mats);
}
