#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <vector>
#include <set>
#include <iostream>
#include "common.h"
using namespace std;

#define CALC_POS(row, col, num_blocks) (row* num_blocks + col)
double GRID_SIZE = 0;
double TOTAL_SIZE = 1;


void apply_force( particle_t &particle, particle_t &neighbor , double *min_dist, double *avg_dist, int *num_counts)
{

    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    if (r2 > cutoff * cutoff)
    {
        return;
    }
	if (r2 != 0)
    {
	   if (r2/(cutoff*cutoff) < *min_dist * (*min_dist))
       {
	    *min_dist = sqrt(r2)/cutoff;
       }

       else
       {
        (*avg_dist) += sqrt(r2)/cutoff;
        (*num_counts)++;
       }
    }

    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);
	
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;

}


// Integrate the ODE
void move(particle_t& p,double size) 
{
    // Slightly simplified Velocity Verlet integration
    // Conserves energy better than explicit Euler method
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x += p.vx * dt;
    p.y += p.vy * dt;

    // Bounce from walls
    while (p.x < 0 || p.x > size) {
        p.x = p.x < 0 ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }

    while (p.y < 0 || p.y > size) {
        p.y = p.y < 0 ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }
}



void calc_force_inside_block(set<int>& block, particle_t* parts,double *min_dist, double *avg_dist, int *num_counts)
{

    for (set<int>::iterator it1 = block.begin(); it1 != block.end(); it1++)
	{
      for (set<int>::iterator it2 = block.begin(); it2 != block.end(); it2++)
	  {
        apply_force(parts[*it1], parts[*it2],min_dist,avg_dist,num_counts);
      }
    }
}

void calc_force_btwn_blocks(set<int>& block1, set<int>& block2, particle_t* parts,double *min_dist, double *avg_dist, int *num_counts)
{
  for (set<int>::iterator it1 = block1.begin(); it1 != block1.end(); it1++)
  {
    for (set<int>::iterator it2 = block2.begin(); it2 != block2.end(); it2++)
	{
      apply_force(parts[*it1],parts[*it2],min_dist,avg_dist,num_counts);
    }
  }

}




void move_block(int i, double prev_x, double prev_y,vector<vector<set<int> > >& grid,particle_t* parts, double GRID_SIZE)
{		

	int numRows = TOTAL_SIZE / GRID_SIZE;
	int numCols = TOTAL_SIZE / GRID_SIZE;

	int block_x_prev = min((int)(prev_x / GRID_SIZE), numRows - 1);
	int block_y_prev = min((int)(prev_y / GRID_SIZE), numCols - 1);

	int block_x = min((int)(parts[i].x / GRID_SIZE), numRows - 1);
	int block_y = min((int)(parts[i].y / GRID_SIZE), numCols - 1);

	if (block_x_prev != block_x || block_y_prev != block_y)
	{

		grid[block_y_prev][block_x_prev].erase(i);
		grid[block_y][block_y].insert(i);
	}
}

void calc_force_grid(vector<vector<set<int> > >& grid,particle_t* parts, double GRID_SIZE, double *min_dist, double *avg_dist,int*num_counts)
{

    int numRows = TOTAL_SIZE / GRID_SIZE;
    int numCols = TOTAL_SIZE / GRID_SIZE;
    for (int i = 0; i < numRows; i++)
	{
		for (int j = 0; j < numCols; j++)
		{
			for (set<int>::iterator it = grid[i][j].begin(); it != grid[i][j].end(); it++)
			{
				parts[*it].ax = parts[*it].ay = 0;
			}


			// right
			if (j != numCols - 1)
			{
				calc_force_btwn_blocks(grid[i][j], grid[i][j+1], parts,min_dist,avg_dist,num_counts);
			}
			//  right bottom
			if  (j != numCols - 1 && i != numRows - 1)
			{
				calc_force_btwn_blocks(grid[i][j], grid[i+1][j+1], parts,min_dist,avg_dist,num_counts);
			}
			//  right top
			if (j != numCols - 1 && i != 0)
			{
				calc_force_btwn_blocks(grid[i][j], grid[i-1][j+1], parts,min_dist,avg_dist,num_counts);
			}
			// left
			if(j != 0)
			{
				calc_force_btwn_blocks(grid[i][j], grid[i][j-1], parts,min_dist,avg_dist,num_counts);
			}
			//  left bottom
			if (j != 0 && i != numRows - 1)
			{
				calc_force_btwn_blocks(grid[i][j], grid[i+1][j-1], parts,min_dist,avg_dist,num_counts);
			}
			//  left top
			if (j != 0 && i != 0)
			{
				calc_force_btwn_blocks(grid[i][j], grid[i-1][j-1], parts,min_dist,avg_dist,num_counts);
			}
			// top
			if (i != 0)
			{
				calc_force_btwn_blocks(grid[i][j], grid[i-1][j], parts,min_dist,avg_dist,num_counts);
			}
			// bottom
			if(i != numRows - 1)
			{
				calc_force_btwn_blocks(grid[i][j], grid[i+1][j], parts,min_dist,avg_dist,num_counts);
			}
			else
			{
				calc_force_inside_block(grid[i][j], parts,min_dist,avg_dist,num_counts);
			}

		}


	}
  
}

void move_parts(particle_t* parts, int num_parts, vector<vector<set<int> > >& grid, double grid_size,double size)
{
    for (int i = 0; i < num_parts; i++)
	{
      double prev_x = parts[i].x;
      double prev_y = parts[i].y;
      move(parts[i],size);
      move_block(i, prev_x, prev_y, grid, parts, grid_size);
    }
}


vector<vector<set<int>>> gen_grid(particle_t* parts, int num_parts, double GRID_SIZE)
{
    int numRols = TOTAL_SIZE / GRID_SIZE;
    int numCols = TOTAL_SIZE / GRID_SIZE;
    vector<vector<set<int> > > grid(numRols, vector< set<int> >(numCols, set<int>()));

    for (int i = 0; i < num_parts; i++){
        int block_x = (int)(parts[i].x / GRID_SIZE);
        int which_block_y = (int)(parts[i].y / GRID_SIZE);
        grid[min(which_block_y, numRols - 1)][min(block_x, numCols - 1)].insert(i);
    }
    for (int i = 0; i < numRols; i++)
	{
      for (int j = 0; j < numCols; j++)
	  {
      }
    }

    return grid;
}

double resize_grid_func(particle_t* parts, int num_parts)
{
  double min_x = 1 << 30;
  double min_y = 1 << 30;
  double max_x = -1;
  double max_y = -1;
  for (int i = 0; i < num_parts; i++)
  {
	min_x = min(parts[i].x, min_x);
	max_x = max(parts[i].x, max_x);
	
	min_y = min(parts[i].y, min_y);
	max_y = max(parts[i].y, max_y);
  }
  double size = max(max_x - min_x, max_y - min_y);
  return size;
}

// Put any static global variables here that you will use throughout the simulation.
void init_simulation(particle_t* parts, int num_parts, double size) 
{
	// You can use this space to initialize data objects that you may need
	// This function will be called once before the algorithm begins
	// Do not do any particle simulation here

}

void simulate_one_step(particle_t* parts, int num_parts, double size) 
{
	int num_counts = 0;
	double avg_dist = 0.0;
	double min_dist = 1.0;


    TOTAL_SIZE = resize_grid_func(parts, num_parts);
    GRID_SIZE = TOTAL_SIZE / ((int)sqrt(num_parts));
    if (GRID_SIZE < 0.01)
      GRID_SIZE = (TOTAL_SIZE / ((int)(ceil(TOTAL_SIZE / cutoff))));
	
	vector<vector<set<int>>> grid = gen_grid(parts, num_parts, GRID_SIZE);

	//compute force within the grid
	calc_force_grid(grid, parts, GRID_SIZE,&avg_dist, &min_dist,&num_counts);

	//move the particles
	move_parts(parts, num_parts, grid, GRID_SIZE,size);
}


