#include "common.h"
#include <mpi.h>
#include <cmath>
#include <algorithm>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
using namespace std;
#define REBIN_ID 1
#define GHOST_ID 2
#define TRUE   1
#define FALSE 0
#define GHOST_REGION (cutoff+2)

const int part_sw = 0;
const int part_s  = 1;
const int part_se = 2;
const int part_w  = 3;
const int part_e  = 4;
const int part_nw = 5;
const int part_n  = 6;
const int part_ne = 7;

const int part_SW = 0;
const int part_S  = 1;
const int part_SE = 2;
const int part_W  = 3;
const int part_E  = 4;
const int part_NW = 5;
const int part_N  = 6;
const int part_NE = 7;

const int NONE = -1;

int ghost_arr[8];
particle_t* ghost_arr_parts[8];
MPI_Request mpi_ghost_reqs[8];
particle_t **move_from;
particle_t *move_to;
int *move_from_num;
MPI_Request mpi_move_from_reqs[8];

double left_x, right_x, bottom_y, top_y;

// Put any static global variables here that you will use throughout the simulation.
//particle modifications
int add_part(particle_t &new_parts, int arr, particle_t *parts, char* part_check)
{
    int index_ins = -1;
    for(int i = 0; i < arr; i++)
    {
        if(part_check[i] == FALSE)
        {
            parts[i] = new_parts;
            part_check[i] = TRUE;
            index_ins = i;
            break;
        }
    }
    return index_ins;
}

void remove_part(int idx, char* part_check)
{
    part_check[idx] = FALSE;
}

bool compare_part_id(particle_t left, particle_t right) 
{
    return left.id < right.id;
}

int select_part(int num_parts, particle_t* parts, particle_t* local_parts, char* part_check, double left_x, double right_x, double bottom_y, double top_y)
{
    int curr_part = 0;
    for(int i = 0; i < num_parts; ++i)
    {
        if((parts[i].x >= left_x) && (parts[i].x < right_x) && (parts[i].y >= bottom_y) && (parts[i].y < top_y))
        {
            local_parts[curr_part] = parts[i];
            part_check[curr_part] = TRUE;

            curr_part++;
        }
    }
    return curr_part;
}

//apply and move
void apply_force(particle_t &particle, particle_t &neighbor)
{
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;
    if( r2 > cutoff*cutoff )
        return;
    r2 = fmax( r2, min_r*min_r );
    double r = sqrt( r2 );
    double coef = ( 1 - cutoff / r ) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

void move(particle_t &p, double size)
{
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x  += p.vx * dt;
    p.y  += p.vy * dt;
    while( p.x < 0 || p.x > size )
    {
        p.x  = p.x < 0 ? -p.x : 2*size-p.x;
        p.vx = -p.vx;
    }
    while( p.y < 0 || p.y > size )
    {
        p.y  = p.y < 0 ? -p.y : 2*size-p.y;
        p.vy = -p.vy;
    }
}

//rebinning particles
void init_move_from(int arr_size)
{
    move_to = new particle_t[arr_size];
    move_from = (particle_t**)malloc(8 * sizeof(particle_t*));
    int* move_from_num = new int[arr_size];

    for(int i = 0; i < 8; i++)
    {
        move_from[i] = new particle_t[arr_size];
    }
}

void free_move_from()
{
    for(int i = 0; i < 8; i++)
    {
        free(move_from[i]);
    }

    free(move_to);
    free(move_from);
    free(move_from_num);
}

void prep_rebin(particle_t* parts, char* part_check, int* num_parts, double left_x, double right_x, double bottom_y, double top_y, int* neighbors)
{
    int num_parts_check = 0;
    int num_parts_remove = 0;
    int curr_pos;
    for(int i = 0; i < 8; i++)
        move_from_num[i] = 0;

    for(int i = 0; num_parts_check < (*num_parts); i++)
    {
        if(part_check[i] == FALSE)
            continue;

        curr_pos = -1;
        if((parts[i].y > top_y) && (parts[i].x < left_x) && (neighbors[part_NW] != -1))
        {
            curr_pos = part_NW;
        }
        else if((parts[i].y > top_y) && (parts[i].x > right_x) && (neighbors[part_NE] != -1))
        {
            curr_pos = part_NE;
        }
        else if((parts[i].y > top_y) && (neighbors[part_N] != -1))
        {
            curr_pos = part_N;
        }
        else if((parts[i].y < bottom_y) && (parts[i].x < left_x) && (neighbors[part_SW] != -1))
        {
            curr_pos = part_SW;
        }
        else if((parts[i].y < bottom_y) && (parts[i].x > right_x) && (neighbors[part_SE] != -1))
        {
            curr_pos = part_SE;
        }
        else if((parts[i].y < bottom_y) && (neighbors[part_S] != -1))
        {
            curr_pos = part_S;
        }
        else if((parts[i].x < left_x) && (neighbors[part_W] != -1))
        {
            curr_pos = part_W;
        }
        else if((parts[i].x > right_x) && (neighbors[part_E] != -1))
        {
            curr_pos = part_E;
        }
        if(curr_pos != -1)
        {
            move_from[curr_pos][move_from_num[curr_pos]] = parts[i];
            move_from_num[curr_pos] += 1;
            remove_part(i, part_check);
            num_parts_remove++;
        }

        num_parts_check++;
    }
    (*num_parts) -= num_parts_remove;
}

//handling rebinning to processors
void send_rebinned_parts(int* neighbors)
{
    int num_reqs = 0;
    for(int i = 0; i < 8; ++i)
    {
        if(neighbors[i] != NONE)
        {
            MPI_Isend ((void*)(move_from[i]), move_from_num[i], PARTICLE, neighbors[i], REBIN_ID, MPI_COMM_WORLD, &(mpi_move_from_reqs[num_reqs]));
            num_reqs++;
        }
    }
}

void receive_rebinned_parts(int* neighbors, int num_neighbors, particle_t* parts, char* part_check, int* num_parts, int tot_arr_size, int sub_arr_size)
{
    MPI_Status status;
    int num_parts_received = 0;

    for(int i = 0; i < 8; i++)
    {
        if(neighbors[i] == -1)
            continue;

        MPI_Recv ((void*)(move_to), sub_arr_size, PARTICLE, neighbors[i], REBIN_ID, MPI_COMM_WORLD, &status); 
        MPI_Get_count(&status, PARTICLE, &num_parts_received);

        for(int j = 0; j < num_parts_received; j++)
        {
            if(add_part(move_to[j], tot_arr_size, parts, part_check) == -1)
            {
                printf("Error: insufficient space\n");
                exit(-1);
            }
        }

        (*num_parts) += num_parts_received;
    }

    MPI_Waitall(num_neighbors, mpi_move_from_reqs, MPI_STATUSES_IGNORE);
}


//ghost particles
void init_ghost_arr(int max_parts)
{
    for(int i = 0; i < 8; ++i)
    {
        ghost_arr_parts[i] = new particle_t[max_parts];
    }
}

void free_ghost_arr()
{
    for(int i = 0; i < 8; ++i)
    {
        free(ghost_arr_parts[i]);
    }
}
void prep_ghost_arr(particle_t *parts, char* part_check, int num_parts, double left_x, double  right_x, double bottom_y, double top_y, int* neighbors)
{
    ghost_arr[part_sw] = 0; ghost_arr[part_s] = 0; ghost_arr[part_se] = 0; ghost_arr[part_w] = 0;
    ghost_arr[part_e] = 0; ghost_arr[part_nw] = 0; ghost_arr[part_n] = 0; ghost_arr[part_ne] = 0;

    int ghost_parts = 0;
    for(int i = 0; ghost_parts < num_parts; ++i)
    {
        if(part_check[i] == FALSE) continue;
        ghost_parts++;

        if(parts[i].x <= (left_x + GHOST_REGION)) // W, SW, or NW ghost zone by x
        {
            if((neighbors[part_w] != -1)) // W neighbor 
            {
                ghost_arr_parts[part_w][ghost_arr[part_w]] = parts[i];
                ++ghost_arr[part_w];
            }
            if(parts[i].y <= (bottom_y + GHOST_REGION)) // SW neighbor  and y bounded
            {
                if((neighbors[part_sw] != -1)) 
                {
                    ghost_arr_parts[part_sw][ghost_arr[part_sw]] = parts[i];
                    ++ghost_arr[part_sw];
                }
            }
            else if (parts[i].y >= (top_y - GHOST_REGION)) // NW neighbor  and y bounded
            {
                if((neighbors[part_nw] != -1)) 
                {
                    ghost_arr_parts[part_nw][ghost_arr[part_nw]] = parts[i];
                    ++ghost_arr[part_nw];
                }
            }
        }
        else if(parts[i].x >= (right_x - GHOST_REGION)) // E, SE, or NE ghost zone by x
        {
            if((neighbors[part_e] != -1)) // E neighbor exists
            {
                ghost_arr_parts[part_e][ghost_arr[part_e]] = parts[i];
                ++ghost_arr[part_e];
            }
            if(parts[i].y <= (bottom_y + GHOST_REGION)) //  SE neighbor  and y bounded
            {
                if((neighbors[part_se] != -1)) 
                {
                    ghost_arr_parts[part_se][ghost_arr[part_se]] = parts[i];
                    ++ghost_arr[part_se];
                }
            }
            else if (parts[i].y >= (top_y - GHOST_REGION)) // NE neighbor  and y bounded
            {
                if((neighbors[part_ne] != -1)) 
                {
                    ghost_arr_parts[part_ne][ghost_arr[part_ne]] = parts[i];
                    ++ghost_arr[part_ne];
                }
            }
        }

        if(parts[i].y <= (bottom_y + GHOST_REGION)) // S ghost zone by y 
        {
            if((neighbors[part_s] != -1)) //  S neighbor 
            {
                ghost_arr_parts[part_s][ghost_arr[part_s]] = parts[i];
                ++ghost_arr[part_s];
            }
        }
        else if(parts[i].y >= (top_y - GHOST_REGION)) // N ghost zone by y 
        {
            if((neighbors[part_n] != -1)) // S neighbor 
            {
                ghost_arr_parts[part_n][ghost_arr[part_n]] = parts[i];
                ++ghost_arr[part_n];
            }
        }
    }
}

void send_ghost_arr(int *neighbors)
{
    int ghost_reqs = 0;
    for(int i = 0; i < 8; ++i)
    {
        if(neighbors[i] != NONE)
        {
            MPI_Isend(ghost_arr_parts[i], ghost_arr[i], PARTICLE, neighbors[i], GHOST_ID, MPI_COMM_WORLD, &(mpi_ghost_reqs[ghost_reqs++]));
        }
    }
}

void receive_ghost_arr(int* num_ghost_parts, particle_t* ghost_parts, int* neighbors, int num_neighbors, int sub_arr_size)
{
    MPI_Status status;
    *num_ghost_parts = 0;

    for(int i = 0; i < 8; i++)
    {
        int num_parts_rec = 0;
        if(neighbors[i] == NONE) continue;

        MPI_Recv(ghost_parts+(*num_ghost_parts), (sub_arr_size-(*num_ghost_parts)), PARTICLE, neighbors[i], GHOST_ID, MPI_COMM_WORLD, &status); 
        MPI_Get_count(&status, PARTICLE, &num_parts_rec);
        *num_ghost_parts += num_parts_rec;
    }

    MPI_Waitall(num_neighbors, mpi_ghost_reqs, MPI_STATUSES_IGNORE);
}


//init params 
particle_t *local;
char *part_check;
int num_local;
int num_ghosts;

particle_t* ghost_parts;
int num_neighbors;
particle_t* parts_tot;
int neighbors[8];


void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) 
{
    // You can use this space to initialize data objects that you may need
    // This function will be called once before the algorithm begins
    // Do not do any particle simulation here

    //divvy up parts to all processors 
    int num_proc_x, num_proc_y;
    for(int stride = (int)floor(sqrt((double)num_procs)); stride >= 1; --stride)
    {
        if(num_procs % stride == 0)
        {
            num_proc_x = stride;
            num_proc_y = num_procs/stride;
            break;
        }
    }
    int proc_x, proc_y;
    proc_x = rank % num_proc_x;
    proc_y = rank/num_proc_x;

    double left_x, right_x, bottom_y, top_y;
    left_x   = (proc_x==0)            ? (0)        : ((size/num_proc_x)*proc_x);
    right_x  = (proc_x==num_proc_x-1) ? (size) : ((size/num_proc_x)*(proc_x+1));
    bottom_y = (proc_y==0)            ? (0)        : ((size/num_proc_y)*proc_y);
    top_y    = (proc_y==num_proc_y-1) ? (size) : ((size/num_proc_y)*(proc_y+1));
    //neighborhoods 
    num_neighbors = 0;
    neighbors[part_sw] = ((proc_x != 0)            && (proc_y != 0)           ) ? ((proc_y-1)*num_proc_x + (proc_x-1)) : (NONE);
    neighbors[part_s ] = (                            (proc_y != 0)           ) ? ((proc_y-1)*num_proc_x + (proc_x  )) : (NONE);
    neighbors[part_se] = ((proc_x != num_proc_x-1) && (proc_y != 0)           ) ? ((proc_y-1)*num_proc_x + (proc_x+1)) : (NONE);

    neighbors[part_w] = ((proc_x != 0)                                       ) ? ((proc_y  )*num_proc_x + (proc_x-1)) : (NONE);
    neighbors[part_e] = ((proc_x != num_proc_x-1)                            ) ? ((proc_y  )*num_proc_x + (proc_x+1)) : (NONE);

    neighbors[part_nw] = ((proc_x != 0)            && (proc_y != num_proc_y-1)) ? ((proc_y+1)*num_proc_x + (proc_x-1)) : (NONE);
    neighbors[part_n] = (                            (proc_y != num_proc_y-1)) ? ((proc_y+1)*num_proc_x + (proc_x  )) : (NONE);
    neighbors[part_ne] = ((proc_x != num_proc_x-1) && (proc_y != num_proc_y-1)) ? ((proc_y+1)*num_proc_x + (proc_x+1)) : (NONE);

    for(int i = 0 ; i < 8; ++i)
    {
        if(neighbors[i] != NONE) num_neighbors++;
    }
    //init params
    local = new particle_t[num_parts];
    part_check = new char[num_parts];
    
    init_ghost_arr(num_parts);
    init_move_from(num_parts);
    
    particle_t* parts_tot = new particle_t[num_parts];
    num_local = select_part(num_parts, parts, local, part_check, left_x, right_x, bottom_y, top_y);
    
    particle_t* ghost_parts = new particle_t[num_parts];
    num_ghosts = 0;
}


void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) 
{
    //ghost parts 
    prep_ghost_arr(local, part_check, num_local, left_x, right_x, bottom_y, top_y, neighbors);
    send_ghost_arr(neighbors);
    receive_ghost_arr(&num_ghosts, ghost_parts, neighbors, num_neighbors, num_parts);
    // Compute Forces
    int spot_parts = 0;
    for(int i = 0; spot_parts < num_parts; ++i)
    {
        if(part_check[i] == FALSE) continue;
        spot_parts++;

        local[i].ax = local[i].ay = 0;
        int nearby_spot_parts = 0;
        for (int j = 0; nearby_spot_parts < num_parts; ++j)
        {
            if(part_check[j] == FALSE) continue;
            nearby_spot_parts++;

            apply_force(local[i], local[j]);
        }

        for(int j = 0; j < num_ghosts; ++j)
        {
            apply_force(local[i], ghost_parts[j]);
        }
    }

    // Move Particles
    for(int i = 0; spot_parts < num_local; ++i)
    {
        if(part_check[i] == FALSE) continue;
        spot_parts++;
        move(local[i],size);

    }
    //rebinning 
    prep_rebin(local, part_check, &num_local, left_x, right_x, bottom_y, top_y, neighbors);
    send_rebinned_parts(neighbors);
    receive_rebinned_parts(neighbors, num_neighbors, local, part_check, &num_local, num_parts, num_parts);
 
}

void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) 
{
    // Write this function such that at the end of it, the master (rank == 0)
    // processor has an in-order view of all particles. That is, the array
    // parts is complete and sorted by particle id.    
    int* node_parts_num    = new int[num_procs];
    int* node_parts_offset = new int[num_procs];
    MPI_Gather(&num_local, 1, MPI_INT, node_parts_num, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if(rank == 0)
    {
        node_parts_offset[0] = 0;
        for(int i = 1; i < num_procs; ++i)
        {
            node_parts_offset[i] = node_parts_offset[i-1] + node_parts_num[i-1];
        }
    }

    particle_t* col_local = new particle_t[num_local];

    int spot_parts = 0;
    for(int i = 0; spot_parts < num_local; ++i)
    {
        if(part_check[i] == FALSE) continue;
        col_local[spot_parts] = local[i];
        spot_parts++;
    }

    MPI_Gatherv(col_local, num_local, PARTICLE, parts, node_parts_num, node_parts_offset, PARTICLE, 0, MPI_COMM_WORLD);
    if(rank == 0)
    {
        sort(parts, parts + num_parts, compare_part_id);
    }
    free(col_local);
    free(node_parts_num);
    free(node_parts_offset);

}