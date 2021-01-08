#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <libgen.h> //Needed for dirname

#include <SDL.h>
#include <SDL_image.h>

char *dir; //Directory path of the current obj file

typedef struct Material{
	int index;
	char name[100];
  	double Ka[3]; //ambient rgb
  	double Kd[3]; //diffuse rgb
  	double Ks[3]; //specular rgb
  	double Ke[3];
  	int illum; //illumination model, ranging from 1-10
  	double Ns; //shininess of material
  	double Tr;
  	double Tf[3];
  	double Ni;
  	double d; //transperency of material
  	SDL_Surface *map_Ka; //ambient texture map
  	SDL_Surface *map_Kd; //diffuse texture map
  	SDL_Surface *map_Ks; //specular texture map
} Material;

typedef struct Triangle{
  double A[3]; //x,y,z of 1st point
  double B[3]; //x,y,z of 2nd point
  double C[3]; //x,y,z of 3rd point
  
  double At[2]; //texture vertex of 1st point
  double Bt[2]; //texture vertex of 2nd point
  double Ct[2]; //texture vertex of 3rd point

  double An[3]; //x,y,z of 1st normal
  double Bn[3]; //x,y,z of 2nd normal
  double Cn[3]; //x,y,z of 3rd normal

  Material mtl; //Material of the triangle
} Triangle;

int num_tris;
int num_mats;
int current_mat = -1;

double x[100000], y[100000], z[100000], xnormal[50000], ynormal[50000], znormal[50000];
double u[50000], v[50000], w[50000];
Triangle tris[75000];
Material mats[100];

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
	num_tris = 0; num_mats = 0;
	int num_v = 0; int num_vt = 0; int num_vn = 0;
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
			//FIRST VERTEX OF TRIANGLE
    		fscanf(f, "%d", &index_vA);

    		tris[num_tris].A[0] = x[index_vA];
    		tris[num_tris].A[1] = y[index_vA];
    		tris[num_tris].A[2] = z[index_vA];

			//SECOND VERTEX OF TRIANGLE
    		fscanf(f, "%d", &index_vB);

    		tris[num_tris].B[0] = x[index_vB];
    		tris[num_tris].B[1] = y[index_vB];
    		tris[num_tris].B[2] = z[index_vB];

			//THIRD VERTEX OF TRIANGLE
    		fscanf(f, "%d", &index_vC);

    		tris[num_tris].C[0] = x[index_vC];
    		tris[num_tris].C[1] = y[index_vC];
    		tris[num_tris].C[2] = z[index_vC];

			//Normal vectors not provided
			//Calculated here, same normal added to all three vertices
    		AB[0] = x[index_vB] - x[index_vA];
    		AB[1] = y[index_vB] - y[index_vA];
    		AB[2] = z[index_vB] - z[index_vA];

    		AC[0] = x[index_vC] - x[index_vA];
    		AC[1] = y[index_vC] - y[index_vA];
    		AC[2] = z[index_vC] - z[index_vA];

    		M3d_x_product(n, AB, AC);

    		tris[num_tris].An[0] = n[0];
    		tris[num_tris].An[1] = n[1];
    		tris[num_tris].An[2] = n[2];

    		tris[num_tris].Bn[0] = n[0];
    		tris[num_tris].Bn[1] = n[1];
    		tris[num_tris].Bn[2] = n[2];

    		tris[num_tris].Cn[0] = n[0];
    		tris[num_tris].Cn[1] = n[1];
    		tris[num_tris].Cn[2] = n[2];

			//Check if face is a quadrilateral
    		if(fscanf(f, "%d", &index_vD) == 1){
	  			//Triangulate the quadrilateral
    			num_tris++;

	  			//FIRST VERTEX OF SECOND TRIANGLE
    			tris[num_tris].A[0] = x[index_vA];
    			tris[num_tris].A[1] = y[index_vA];
    			tris[num_tris].A[2] = z[index_vA];

	  			//SECOND VERTEX OF SECOND TRIANGLE
    			tris[num_tris].B[0] = x[index_vC];
    			tris[num_tris].B[1] = y[index_vC];
    			tris[num_tris].B[2] = z[index_vC];

	  			//THIRD VERTEX OF SECOND TRIANGLE
    			tris[num_tris].C[0] = x[index_vD];
    			tris[num_tris].C[1] = y[index_vD];
    			tris[num_tris].C[2] = z[index_vD];

	  			//Normal vector for the vertices not provided by file
	  			//Normal vector for surface is the same as 1st triangle
    			tris[num_tris].An[0] = n[0];
    			tris[num_tris].An[1] = n[1];
    			tris[num_tris].An[2] = n[2];

    			tris[num_tris].Bn[0] = n[0];
    			tris[num_tris].Bn[1] = n[1];
    			tris[num_tris].Bn[2] = n[2];

    			tris[num_tris].Cn[0] = n[0];
    			tris[num_tris].Cn[1] = n[1];
    			tris[num_tris].Cn[2] = n[2];
    		}
    	}
      else if(format == 2){//Face format: v/vt v/vt v/vt
		//FIRST VERTEX OF TRIANGLE
      	fscanf(f, "%d/%d", &index_vA, &index_vt);

      	tris[num_tris].A[0] = x[index_vA];
      	tris[num_tris].A[1] = y[index_vA];
      	tris[num_tris].A[2] = z[index_vA];

      	tris[num_tris].At[0] = u[index_vt];
      	tris[num_tris].At[1] = v[index_vt];

		//SECOND VERTEX OF TRIANGLE
      	fscanf(f, "%d/%d", &index_vB, &index_vt);

      	tris[num_tris].B[0] = x[index_vB];
      	tris[num_tris].B[1] = y[index_vB];
      	tris[num_tris].B[2] = z[index_vB];

      	tris[num_tris].Bt[0] = u[index_vt];
      	tris[num_tris].Bt[1] = v[index_vt];

		//THIRD VERTEX OF TRIANGLE
      	fscanf(f, "%d/%d", &index_vC, &index_vt);

      	tris[num_tris].C[0] = x[index_vC];
      	tris[num_tris].C[1] = y[index_vC];
      	tris[num_tris].C[2] = z[index_vC];

      	tris[num_tris].Ct[0] = u[index_vt];
      	tris[num_tris].Ct[1] = v[index_vt];

		//Normal vectors not provided
		//Calculated here, same normal added to all three vertices
      	AB[0] = x[index_vB] - x[index_vA];
      	AB[1] = y[index_vB] - y[index_vA];
      	AB[2] = z[index_vB] - z[index_vA];

      	AC[0] = x[index_vC] - x[index_vA];
      	AC[1] = y[index_vC] - y[index_vA];
      	AC[2] = z[index_vC] - z[index_vA];

      	M3d_x_product(n, AB, AC);

      	tris[num_tris].An[0] = n[0];
      	tris[num_tris].An[1] = n[1];
      	tris[num_tris].An[2] = n[2];

      	tris[num_tris].Bn[0] = n[0];
      	tris[num_tris].Bn[1] = n[1];
      	tris[num_tris].Bn[2] = n[2];

      	tris[num_tris].Cn[0] = n[0];
      	tris[num_tris].Cn[1] = n[1];
      	tris[num_tris].Cn[2] = n[2];

		//Check if face is a quadrilateral
      	if(fscanf(f, "%d/%d", &index_vD, &index_vt) == 2){
	  		//Triangulate the quadrilateral
      		num_tris++;

	  		//FIRST VERTEX OF SECOND TRIANGLE
      		tris[num_tris].A[0] = x[index_vA];
      		tris[num_tris].A[1] = y[index_vA];
      		tris[num_tris].A[2] = z[index_vA];

	  		//Shares texture vertices with first vertex of 1st triangle
      		tris[num_tris].At[0] = tris[num_tris-1].At[0];
      		tris[num_tris].At[1] = tris[num_tris-1].At[1];

	  		//SECOND VERTEX OF SECOND TRIANGLE
      		tris[num_tris].B[0] = x[index_vC];
      		tris[num_tris].B[1] = y[index_vC];
      		tris[num_tris].B[2] = z[index_vC];

	  		//Shares texture vertices with third vertex of 1st triangle
      		tris[num_tris].Bt[0] = tris[num_tris-1].Ct[0];
      		tris[num_tris].Bt[1] = tris[num_tris-1].Ct[1];

	  		//THIRD VERTEX OF SECOND TRIANGLE
      		tris[num_tris].C[0] = x[index_vD];
      		tris[num_tris].C[1] = y[index_vD];
      		tris[num_tris].C[2] = z[index_vD];

      		tris[num_tris].Ct[0] = u[index_vt];
      		tris[num_tris].Ct[1] = v[index_vt];


	  		//Normals vector for the vertices not provided by file
	  		//Normal vector for surface is the same as 1st triangle
      		tris[num_tris].An[0] = n[0];
      		tris[num_tris].An[1] = n[1];
      		tris[num_tris].An[2] = n[2];

      		tris[num_tris].Bn[0] = n[0];
      		tris[num_tris].Bn[1] = n[1];
      		tris[num_tris].Bn[2] = n[2];

      		tris[num_tris].Cn[0] = n[0];
      		tris[num_tris].Cn[1] = n[1];
      		tris[num_tris].Cn[2] = n[2];
      	}
      }
      else if(format == 3){//Face format: v/vt/vn  v/vt/vn v/vt/vn
		//FIRST VERTEX OF TRIANGLE
      	fscanf(f, "%d/%d/%d", &index_vA, &index_vt, &index_vn);

      	tris[num_tris].A[0] = x[index_vA];
      	tris[num_tris].A[1] = y[index_vA];
      	tris[num_tris].A[2] = z[index_vA];

      	tris[num_tris].At[0] = u[index_vt];
      	tris[num_tris].At[1] = v[index_vt];

      	tris[num_tris].An[0] = xnormal[index_vn];
      	tris[num_tris].An[1] = ynormal[index_vn];
      	tris[num_tris].An[2] = znormal[index_vn];

		//SECOND VERTEX OF TRIANGLE
      	fscanf(f, "%d/%d/%d", &index_vB, &index_vt, &index_vn);

      	tris[num_tris].B[0] = x[index_vB];
      	tris[num_tris].B[1] = y[index_vB];
      	tris[num_tris].B[2] = z[index_vB];

      	tris[num_tris].Bt[0] = u[index_vt];
      	tris[num_tris].Bt[1] = v[index_vt];

      	tris[num_tris].Bn[0] = xnormal[index_vn];
      	tris[num_tris].Bn[1] = ynormal[index_vn];
      	tris[num_tris].Bn[2] = znormal[index_vn];

		//THIRD VERTEX OF TRIANGLE
      	fscanf(f, "%d/%d/%d", &index_vC, &index_vt, &index_vn);

      	tris[num_tris].C[0] = x[index_vC];
      	tris[num_tris].C[1] = y[index_vC];
      	tris[num_tris].C[2] = z[index_vC];

      	tris[num_tris].Ct[0] = u[index_vt];
      	tris[num_tris].Ct[1] = v[index_vt];

      	tris[num_tris].Cn[0] = xnormal[index_vn];
      	tris[num_tris].Cn[1] = ynormal[index_vn];
      	tris[num_tris].Cn[2] = znormal[index_vn];

	//Check if face is a quadrilateral
      	if(fscanf(f, "%d/%d/%d", &index_vD, &index_vt, &index_vn) == 3){
	  //Triangulate the quadrilateral
      		num_tris++;

	  //FIRST VERTEX OF SECOND TRIANGLE
      		tris[num_tris].A[0] = x[index_vA];
      		tris[num_tris].A[1] = y[index_vA];
      		tris[num_tris].A[2] = z[index_vA];

	  //Shares texture vertices with first vertex of 1st triangle
      		tris[num_tris].At[0] = tris[num_tris-1].At[0];
      		tris[num_tris].At[1] = tris[num_tris-1].At[1];

      		tris[num_tris].An[0] = tris[num_tris-1].An[0];
      		tris[num_tris].An[1] = tris[num_tris-1].An[1];
      		tris[num_tris].An[2] = tris[num_tris-1].An[2];

	  //SECOND VERTEX OF SECOND TRIANGLE
      		tris[num_tris].B[0] = x[index_vC];
      		tris[num_tris].B[1] = y[index_vC];
      		tris[num_tris].B[2] = z[index_vC];

	  //Shares texture vertices with third vertex of 1st triangle
      		tris[num_tris].Bt[0] = tris[num_tris-1].Ct[0];
      		tris[num_tris].Bt[1] = tris[num_tris-1].Ct[1];

      		tris[num_tris].Bn[0] = tris[num_tris-1].Cn[0];
      		tris[num_tris].Bn[1] = tris[num_tris-1].Cn[1];
      		tris[num_tris].Bn[2] = tris[num_tris-1].Cn[2];

	  //THIRD VERTEX OF SECOND TRIANGLE
      		tris[num_tris].C[0] = x[index_vD];
      		tris[num_tris].C[1] = y[index_vD];
      		tris[num_tris].C[2] = z[index_vD];

      		tris[num_tris].Ct[0] = u[index_vt];
      		tris[num_tris].Ct[1] = v[index_vt];

      		tris[num_tris].Cn[0] = xnormal[index_vn];
      		tris[num_tris].Cn[1] = ynormal[index_vn];
      		tris[num_tris].Cn[2] = znormal[index_vn];

      	}
      }
      else if(format == 4){//Face format: v//vn v//vn v//vn
	//FIRST VERTEX OF TRIANGLE
      	fscanf(f, "%d//%d", &index_vA, &index_vn);


      	tris[num_tris].A[0] = x[index_vA];
      	tris[num_tris].A[1] = y[index_vA];
      	tris[num_tris].A[2] = z[index_vA];

      	tris[num_tris].An[0] = xnormal[index_vn];
      	tris[num_tris].An[1] = ynormal[index_vn];
      	tris[num_tris].An[2] = znormal[index_vn];

	//SECOND VERTEX OF TRIANGLE
      	fscanf(f, "%d//%d", &index_vB, &index_vn);

      	tris[num_tris].B[0] = x[index_vB];
      	tris[num_tris].B[1] = y[index_vB];
      	tris[num_tris].B[2] = z[index_vB];

      	tris[num_tris].Bn[0] = xnormal[index_vn];
      	tris[num_tris].Bn[1] = ynormal[index_vn];
      	tris[num_tris].Bn[2] = znormal[index_vn];

	//THIRD VERTEX OF TRIANGLE
      	fscanf(f, "%d//%d", &index_vC, &index_vn);

      	tris[num_tris].C[0] = x[index_vC];
      	tris[num_tris].C[1] = y[index_vC];
      	tris[num_tris].C[2] = z[index_vC];

      	tris[num_tris].Cn[0] = xnormal[index_vn];
      	tris[num_tris].Cn[1] = ynormal[index_vn];
      	tris[num_tris].Cn[2] = znormal[index_vn];

	//Check if face is a quadrilateral
      	if(fscanf(f, "%d//%d", &index_vD, &index_vn) == 2){
	  //Triangulate the quadrilateral
      		num_tris++;

	  //FIRST VERTEX OF SECOND TRIANGLE
      		tris[num_tris].A[0] = x[index_vA];
      		tris[num_tris].A[1] = y[index_vA];
      		tris[num_tris].A[2] = z[index_vA];

      		tris[num_tris].An[0] = tris[num_tris-1].An[0];
      		tris[num_tris].An[1] = tris[num_tris-1].An[1];
      		tris[num_tris].An[2] = tris[num_tris-1].An[2];

	  //SECOND VERTEX OF SECOND TRIANGLE
      		tris[num_tris].B[0] = x[index_vC];
      		tris[num_tris].B[1] = y[index_vC];
      		tris[num_tris].B[2] = z[index_vC];

      		tris[num_tris].Bn[0] = tris[num_tris-1].Cn[0];
      		tris[num_tris].Bn[1] = tris[num_tris-1].Cn[1];
      		tris[num_tris].Bn[2] = tris[num_tris-1].Cn[2];

	  //THIRD VERTEX OF SECOND TRIANGLE
      		tris[num_tris].C[0] = x[index_vD];
      		tris[num_tris].C[1] = y[index_vD];
      		tris[num_tris].C[2] = z[index_vD];

      		tris[num_tris].Cn[0] = xnormal[index_vn];
      		tris[num_tris].Cn[1] = ynormal[index_vn];
      		tris[num_tris].Cn[2] = znormal[index_vn];
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
free(x);
free(y);
SDL_Log("Triangles: %d\n", num_tris);
SDL_Log("Finished loading %s\n", fname);
return 1;
}
