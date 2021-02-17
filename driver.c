#include <stdio.h>
#include <math.h>

#include <SDL.h>
#include <SDL_image.h>

#include "M3d_matrix_tools.h"
#include "obj_reader.h"
#include "light_model.h"
#include "graphic_tools.h"
#include "kernel.h"

int main(int argc, char** argv)
{
    init_graphics();
    init_IMG();

    double degrees_of_half_angle;

    degrees_of_half_angle = 30;
    read_obj_file("./objects/teapot/teapot.obj");

    double tan_half = tan(degrees_of_half_angle * M_PI / 180);

    int s, e, frame_number;
    s = 0; e = 15;
    for (frame_number = s; frame_number < e; frame_number++) {
        SDL_Log("frame: %d\n", frame_number);
        set_rgb(0, 0, 0);
        SDL_RenderClear(S_Renderer);

        double eangle = 2 * M_PI * frame_number / e;
        double eye[3];

        //Location of eye (world space)
        eye[0] = 15.0 * cos(eangle);
        eye[1] = 10.0;
        eye[2] = 15.0 * sin(eangle);

        double coi[3];

        //Center of interest/where the eye is looking (world space)
        coi[0] = 0;
        coi[1] = 0;
        coi[2] = 0;

        double up[3];

        //What direction is up for the eye
        up[0] = eye[0];
        up[1] = eye[1] + 1;
        up[2] = eye[2];

        double vm[4][4], vi[4][4];
        M3d_view(vm, vi, eye, coi, up);

        double light_in_world_space[3];

        num_lights = 0;
        light_rgb[num_lights][0] = 0.8;
        light_rgb[num_lights][1] = 0.0;
        light_rgb[num_lights][2] = 0.0;
        
        light_in_world_space[0] = 300;
        light_in_world_space[1] = 200;
        light_in_world_space[2] = 300;
        M3d_mat_mult_pt(light_in_eye_space[num_lights], vm, light_in_world_space);
        num_lights++;

        light_rgb[num_lights][0] = 0.1;
        light_rgb[num_lights][1] = 0.0;
        light_rgb[num_lights][2] = 1.0;

        light_in_world_space[0] = -300;
        light_in_world_space[1] = -200;
        light_in_world_space[2] = -300;
        M3d_mat_mult_pt(light_in_eye_space[num_lights], vm, light_in_world_space);
        num_lights++;

        double Ka[3], Kd[3], Ks[3];

        // Transform the object :
        double Tvlist[100];
        int Tn, Ttypelist[100];
        double m[4][4], mi[4][4];
        double obmat[4][4], obinv[4][4];

        Tn = 0;
        Ttypelist[Tn] = RX; Tvlist[Tn] = 270; Tn++;

        M3d_make_movement_sequence_matrix(m, mi, Tn, Ttypelist, Tvlist);
        M3d_mat_mult(obmat, vm, m);
        M3d_mat_mult(obinv, mi, vi);

        //Transforming vertices and normals from object space to eye space
        mat_mult_device(x, y, z, obmat, x, y, z, num_v + 1);



        ///////////////////////////////////////////////////////////////////
        ///////////////////////////////////////////////////////////////////
        ///////////////////////////////////////////////////////////////////
        double origin[3], screen_pt[3];
        double uv[2], point[3], normal[3];
        double argb[3];

        origin[0] = 0;
        origin[1] = 0;
        origin[2] = 0;

        int x_pix, y_pix;
        for (x_pix = 0; x_pix < SCREEN_WIDTH; x_pix++)
        {
            SDL_Log("x: %d\n", x_pix);
            for (y_pix = 0; y_pix < SCREEN_HEIGHT; y_pix++)
            {
                screen_pt[0] = x_pix - SCREEN_WIDTH / 2;
                screen_pt[1] = y_pix - SCREEN_HEIGHT / 2;
                screen_pt[2] = (SCREEN_WIDTH / 2) / tan_half;

                s = intersect_all_triangles_device(origin, screen_pt, uv, point);
                if (s == -1)
                {
                    argb[0] = argb[1] = argb[2] = 0.5;
                }
                else
                {
                    interpolate_normal_vector(s, uv, obinv, normal);
                    if (tris[s].mtl.index != -1)
                    {
                        
                        if (tris[s].mtl.map_Ka)
                        {
                            get_rgb(tris[s].mtl.map_Ka, tris[s].At, tris[s].Bt, tris[s].Ct, uv, Ka);
                        }
                        else {
                            Ka[0] = tris[s].mtl.Ka[0];
                            Ka[1] = tris[s].mtl.Ka[1];
                            Ka[2] = tris[s].mtl.Ka[2];
                        }

                        if (tris[s].mtl.map_Kd)
                        {

                            get_rgb(tris[s].mtl.map_Kd, tris[s].At, tris[s].Bt, tris[s].Ct, uv, Kd);
                        }
                        else {
                            Kd[0] = tris[s].mtl.Kd[0];
                            Kd[1] = tris[s].mtl.Kd[1];
                            Kd[2] = tris[s].mtl.Kd[2];
                        }

                        if (tris[s].mtl.map_Ks)
                        {
                            get_rgb(tris[s].mtl.map_Ks, tris[s].At, tris[s].Bt, tris[s].Ct, uv, Ks);
                        }
                        else {
                            Ks[0] = tris[s].mtl.Ks[0];
                            Ks[1] = tris[s].mtl.Ks[1];
                            Ks[2] = tris[s].mtl.Ks[2];
                        }
                    }
                    else
                    {
                        Ka[0] = 1.0; Ka[1] = 1.0; Ka[2] = 1.0;
                        Kd[0] = 1.0; Kd[1] = 1.0; Kd[2] = 1.0;
                        Ks[0] = 1.0; Ks[1] = 1.0; Ks[2] = 1.0;
                    }
                    Light_Model(Ka, Kd, Ks, origin, point, normal, argb);
                }

                int screen_y;
                screen_y = SCREEN_HEIGHT - y_pix;

                set_rgb(argb[0], argb[1], argb[2]);
                SDL_RenderDrawPoint(S_Renderer, x_pix, screen_y);

            } // end for y_pix
        } // end for x_pix

        SDL_RenderPresent(S_Renderer);
        char fname[200];
        sprintf(fname, "pic/pic%04d.bmp", frame_number);

        save_image_to_file(fname);

        //Transforming vertices and normals back to object space
        mat_mult_device(x, y, z, obinv, x, y, z, num_v + 1);
    } // end for frame_number

    close_graphics();
    close_object();

    return 1;
}
