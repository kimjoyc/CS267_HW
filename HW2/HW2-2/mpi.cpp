#include <unistd.h>
#include <stdio.h>
#include <assert.h>
#include "common.h"
#include <mpi.h>
#include <cmath>
#include <algorithm>
#include <vector>
#include <unordered_set>

// Put any static global variables here that you will use throughout the simulation.

const double grid_step = cutoff*1.0001;
int g_lda;
int g_dims[2], g_coords[2], g_xd, g_yd, g_x0, g_y0;
MPI_Comm g_comm;

struct ParticleContainer;
struct ParticleContainerPredecessor;
struct ParticleContainerPredecessor {
    ParticleContainer* next = nullptr;
};
struct ParticleContainer : ParticleContainerPredecessor {
    ParticleContainerPredecessor* prev = nullptr;
    particle_t* p = nullptr;
    int g_i;
};
ParticleContainerPredecessor* particle_grid;
ParticleContainer* particle_containers;
std::unordered_set<int> g_parts, g_ghosts;
std::vector<particle_t> x_send_down, x_send_up, y_send_down, y_send_up,
    x_recv_down, x_recv_up, y_recv_down, y_recv_up;

static void inline __attribute__((always_inline))
linkParticle(ParticleContainerPredecessor* pred, ParticleContainer* pc) {
    pc->prev = pred;
    pc->next = pred->next;
    if (pred->next != nullptr)
        pred->next->prev = pc;
    pred->next = pc;
}

static void inline __attribute__((always_inline))
unlinkParticle(ParticleContainer* pc) {
    pc->prev->next = pc->next;
    if (pc->next != nullptr)
        pc->next->prev = pc->prev;
}

// Apply the force from neighbor to particle
static void inline __attribute__((always_inline))
apply_force(particle_t& p, particle_t& p_prime) {
    // Calculate Distance
    double dx = p_prime.x - p.x;
    double dy = p_prime.y - p.y;
    double r2 = dx * dx + dy * dy;

    // Check if the two particles should interact
    if (r2 > cutoff * cutoff)
        return;

    r2 = std::max(r2, min_r * min_r);
    double r = sqrt(r2);

    // Very simple short-range repulsive force
    double coef = (1 - cutoff / r) / r2 / mass;
    p.ax += coef * dx;
    p.ay += coef * dy;
    p_prime.ax -= coef * dx;
    p_prime.ay -= coef * dy;
}

static void inline __attribute__((always_inline))
apply_intercell_force(particle_t* const p, const int x_prime, const int y_prime) {
    int i_prime = x_prime+(g_xd+2)*y_prime;
    for(ParticleContainer* pc_prime = particle_grid[i_prime].next; pc_prime != nullptr; pc_prime = pc_prime->next) {
        particle_t* p_prime = pc_prime->p;
        apply_force(*p, *p_prime);
    }
}

// Integrate the ODE
static void inline __attribute__((always_inline))
move(particle_t& p, double size) {
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

void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) 
{
    // You can use this space to initialize data objects that you may need
    // This function will be called once before the algorithm begins
    // Do not do any particle simulation here
    MPI_Dims_create(num_procs, 2, g_dims);
    int periods[2] = {false, false};
    MPI_Cart_create(MPI_COMM_WORLD, 2, g_dims, periods, false, &g_comm);
    MPI_Cart_coords(g_comm, rank, 2, g_coords);
    g_lda = static_cast<int>(size/grid_step)+1;
    g_xd = (g_lda+g_dims[0]-1)/g_dims[0];
    g_yd = (g_lda+g_dims[1]-1)/g_dims[1];
    g_x0 = g_coords[0]*g_xd;
    g_y0 = g_coords[1]*g_yd;

    particle_grid = new ParticleContainerPredecessor[(g_xd+2)*(g_yd+2)]();
    particle_containers = new ParticleContainer[num_parts]();

    for (int p_i = 0; p_i < num_parts; ++p_i) {
        ParticleContainer* pc = particle_containers+p_i;
        particle_t* p = parts+p_i;
        pc->p = p;
        p->ax = p->ay = 0;
        int g_x = static_cast<int>(p->x / grid_step)-g_x0;
        int g_y = static_cast<int>(p->y / grid_step)-g_y0;
        if (0 <= g_x && g_x < g_xd && 0 <= g_y && g_y < g_yd) {
            g_parts.insert(p_i);
            int g_i = g_x+1+(g_xd+2)*(g_y+1);
            pc->g_i = g_i;
            ParticleContainerPredecessor* g = particle_grid+g_i;
            linkParticle(g, pc);
        }
    }
}

static void inline __attribute__((always_inline))
shift_exchange(
    int dimension, int displacement,
    std::vector<particle_t>& send,
    std::vector<particle_t>& recv,
    int x_lo, int x_hi,
    int y_lo, int y_hi)
{
    int incoming, outgoing;
    MPI_Cart_shift(g_comm, dimension, displacement, &incoming, &outgoing);
    for (int gy = y_lo; gy < y_hi; ++gy) {
      for (int gx = x_lo; gx < x_hi; ++gx) {
        int i = gx+(g_xd+2)*gy;
        for (ParticleContainer* pc = particle_grid[i].next; pc != nullptr; pc = pc->next) {
            particle_t* p = pc->p;
            send.push_back(*p);
        }
      }
    }
    MPI_Send(send.data(), send.size(), PARTICLE, outgoing, 0, g_comm);
    MPI_Status recv_status;
    int recv_count;
    MPI_Probe(incoming, MPI_ANY_TAG, g_comm, &recv_status);
    MPI_Get_count(&recv_status, PARTICLE, &recv_count);
    int recv_offset = recv.size();
    recv.resize(recv_offset+recv_count);
    MPI_Recv(recv.data()+recv_offset, recv_count, PARTICLE, incoming, MPI_ANY_TAG, g_comm, MPI_STATUS_IGNORE);
}

static void inline __attribute__((always_inline))
receive_particles(
    particle_t* parts, ParticleContainer* particle_containers, ParticleContainerPredecessor* particle_grid,
    std::vector<particle_t>& recv)
{
    for (const auto& p_recv: recv) {
        int p_i = p_recv.id - 1;
        ParticleContainer* pc = particle_containers+p_i;
        particle_t* p = parts+p_i;
        *p = p_recv;
        pc->p = p;
        int g_x = static_cast<int>(p->x / grid_step)-g_x0;
        int g_y = static_cast<int>(p->y / grid_step)-g_y0;
        if (-1 <= g_x && g_x < g_xd+1 && -1 <= g_y && g_y < g_yd+1) {
            if (0 <= g_x && g_x < g_xd && 0 <= g_y && g_y < g_yd) {
                g_parts.insert(p_i);
            } else {
                g_ghosts.insert(p_i);
            }
            int g_i = g_x+1+(g_xd+2)*(g_y+1);
            pc->g_i = g_i;
            ParticleContainerPredecessor* g = particle_grid+g_i;
            linkParticle(g, pc);
        }
    }
    recv.resize(0);
}

void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) 
{
MPI_Barrier(g_comm);
    shift_exchange(0, -1, x_send_down, x_recv_up, 1, 2, 1, g_yd+1);
MPI_Barrier(g_comm);
    shift_exchange(0, 1, x_send_up, x_recv_down, g_xd, g_xd+1, 1, g_yd+1);
MPI_Barrier(g_comm);
    shift_exchange(1, -1, y_send_down, y_recv_up, 1, g_xd+1, 1, 2);
MPI_Barrier(g_comm);
    shift_exchange(1, -1, x_recv_down, y_recv_up, 0, 0, 0, 0);
MPI_Barrier(g_comm);
    shift_exchange(1, -1, x_recv_up, y_recv_up, 0, 0, 0, 0);
MPI_Barrier(g_comm);
    shift_exchange(1, 1, y_send_up, y_recv_down, 1, g_xd+1, g_yd, g_yd+1);
MPI_Barrier(g_comm);
    shift_exchange(1, 1, x_recv_down, y_recv_down, 0, 0, 0, 0);
MPI_Barrier(g_comm);
    shift_exchange(1, 1, x_recv_up, y_recv_down, 0, 0, 0, 0);
MPI_Barrier(g_comm);
    receive_particles(parts, particle_containers, particle_grid, x_recv_down);
MPI_Barrier(g_comm);
    receive_particles(parts, particle_containers, particle_grid, x_recv_up);
MPI_Barrier(g_comm);
    receive_particles(parts, particle_containers, particle_grid, y_recv_down);
MPI_Barrier(g_comm);
    receive_particles(parts, particle_containers, particle_grid, y_recv_up);
MPI_Barrier(g_comm);
    x_send_down.resize(0); x_send_up.resize(0);
    y_send_down.resize(0); y_send_up.resize(0);
    // Compute Forces
    for (int gy = 1; gy < g_yd+1; ++gy) {
      for (int gx = 1; gx < g_xd+1; ++gx) {
        int i = gx+(g_xd+2)*gy;
        for (ParticleContainer* pc = particle_grid[i].next; pc != nullptr; pc = pc->next) {
            particle_t* p = pc->p;
            apply_intercell_force(p, gx-1, gy-1);
            apply_intercell_force(p, gx, gy-1);
            apply_intercell_force(p, gx+1, gy-1);
            apply_intercell_force(p, gx-1, gy);
//            apply_intercell_force(p, gx, gy);
//            apply_intercell_force(p, gx+1, gy);
//            apply_intercell_force(p, gx-1, gy+1);
//            apply_intercell_force(p, gx, gy+1);
//            apply_intercell_force(p, gx+1, gy+1);
            for (ParticleContainer* pc_prime = pc->next; pc_prime != nullptr; pc_prime = pc_prime->next) apply_force(*p, *pc_prime->p);
        }
      }
    }
MPI_Barrier(g_comm);
    // Move Particles
    for (auto it = g_parts.begin(); it != g_parts.end(); ) {
        int p_i = *it;
        ParticleContainer* pc = particle_containers+p_i;
        particle_t* p = parts+p_i;
        move(*p, size);
        p->ax = p->ay = 0;
        int g_x_prime = static_cast<int>(p->x / grid_step)-g_x0;
        int g_y_prime = static_cast<int>(p->y / grid_step)-g_y0;
        int g_i_prime = g_x_prime+1+(g_xd+2)*(g_y_prime+1);
        if (pc->g_i != g_i_prime) {
            if (g_x_prime < 0 || g_x_prime >= g_xd || g_y_prime < 0 || g_y_prime >= g_yd) {
              unlinkParticle(pc);
              it = g_parts.erase(it);
              if (g_x_prime < 0) {
                x_send_down.push_back(*p);
              } else if (g_x_prime >= g_xd) {
                x_send_up.push_back(*p);
              } else if (g_y_prime < 0) {
                y_send_down.push_back(*p);
              } else if (g_y_prime >= g_yd) {
                y_send_up.push_back(*p);
              }
              continue;
            }
            ParticleContainerPredecessor* g_prime = particle_grid+g_i_prime;
            pc->g_i = g_i_prime;
            unlinkParticle(pc);
            linkParticle(g_prime, pc);
        }
        ++it;
    }
    for (const int p_i: g_ghosts) {
        unlinkParticle(particle_containers+p_i);
    }
    g_ghosts.clear();
MPI_Barrier(g_comm);
}

void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Write this function such that at the end of it, the master (rank == 0)
    // processor has an in-order view of all particles. That is, the array
    // parts is complete and sorted by particle id.
MPI_Barrier(g_comm);
    std::vector<particle_t> send;
    send.reserve(g_parts.size());
    std::vector<particle_t> recv;
    for (const int p_i: g_parts) {
        send.push_back(parts[p_i]);
    }
    int sendcount = send.size();
    int* recvcounts, *recvdispls;
    if (rank == 0) {
        recvcounts = new int[num_procs];
        recvdispls = new int[num_procs];
    }
    MPI_Barrier(g_comm);
    MPI_Gather(&sendcount, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, g_comm);
    if (rank == 0) {
        recvdispls[0] = 0;
        for (int i = 1; i < num_procs; ++i) recvdispls[i] = recvdispls[i-1]+recvcounts[i-1];
        recv.resize(recvdispls[num_procs-1]+recvcounts[num_procs-1]);
    }
    MPI_Barrier(g_comm);
    MPI_Gatherv(send.data(), sendcount, PARTICLE, recv.data(), recvcounts, recvdispls, PARTICLE, 0, g_comm);
    if (rank == 0) {
        for (const auto& p_recv: recv) {
            int p_i = p_recv.id - 1;
            parts[p_i] = p_recv;
        }
    }
}
