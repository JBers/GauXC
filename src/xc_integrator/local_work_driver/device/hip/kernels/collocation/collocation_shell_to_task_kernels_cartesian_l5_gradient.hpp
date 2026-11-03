/**
 * GauXC Copyright (c) 2020-2024, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy).
 *
 * (c) 2024-2025, Microsoft Corporation
 *
 * All rights reserved.
 *
 * See LICENSE.txt for details
 */
#include "hip/hip_runtime.h"
#pragma once
#include "collocation_device_constants.hpp"
#include "device/xc_device_task.hpp"
#include "device_specific/hip_device_constants.hpp"
#include "device/common/shell_to_task.hpp"
#include <cassert>

namespace GauXC {


__global__ __launch_bounds__(128,2) void collocation_device_shell_to_task_kernel_cartesian_gradient_5(
  uint32_t                        nshell,
  ShellToTaskDevice* __restrict__ shell_to_task,
  XCDeviceTask*      __restrict__ device_tasks
) {


  __shared__ double alpha[128/hip::warp_size][detail::shell_nprim_max + 1]; 
  __shared__ double coeff[128/hip::warp_size][detail::shell_nprim_max + 1];
  double* my_alpha = alpha[threadIdx.x/hip::warp_size];
  double* my_coeff = coeff[threadIdx.x/hip::warp_size];

  for( auto ish = blockIdx.z; ish < nshell; ish += gridDim.z ) {
  const uint32_t ntasks      = shell_to_task[ish].ntask;
  const auto shell           = shell_to_task[ish].shell_device;
  const auto task_idx        = shell_to_task[ish].task_idx_device;
  const auto task_shell_offs = shell_to_task[ish].task_shell_offs_device;


  // Load Shell Data into registers / SM
  const uint32_t nprim = shell->nprim();
  const double3 O  = *reinterpret_cast<const double3*>(shell->O_data());

  const int global_warp_id = (threadIdx.x + blockIdx.x*blockDim.x) / hip::warp_size;
  const int nwarp_global   = max((blockDim.x*gridDim.x) / hip::warp_size,1);

  // Read in coeffs/exps into SM on first warp
  {
    auto* coeff_gm = shell->coeff_data();
    auto* alpha_gm = shell->alpha_data();
    static_assert( detail::shell_nprim_max <= hip::warp_size );
    const int warp_rank = threadIdx.x % hip::warp_size;
    if( warp_rank < detail::shell_nprim_max ) {
      my_alpha[warp_rank] = alpha_gm[warp_rank];
      my_coeff[warp_rank] = coeff_gm[warp_rank];
    }
  }

  // Loop over tasks assigned to shells
  // Place each task on a different warp + schedule across blocks
  for( int itask = global_warp_id; itask < ntasks; itask += nwarp_global ) {

    const auto*              task   = device_tasks + task_idx[itask];
    const auto* __restrict__ points_x = task->points_x;
    const auto* __restrict__ points_y = task->points_y;
    const auto* __restrict__ points_z = task->points_z;
    const uint32_t           npts   = task->npts;
    const size_t             shoff  = task_shell_offs[itask] * npts;

    auto* __restrict__ basis_eval = task->bf + shoff;
    auto* __restrict__ basis_x_eval = task->dbfx + shoff;
    auto* __restrict__ basis_y_eval = task->dbfy + shoff;
    auto* __restrict__ basis_z_eval = task->dbfz + shoff;

    // Loop over points in task
    // Assign each point to separate thread within the warp
    #pragma unroll 1
    for( int ipt = threadIdx.x % hip::warp_size; ipt < npts; ipt += hip::warp_size ) {
      //const double3 point = points[ipt];
      double3 point;
      point.x = points_x[ipt];
      point.y = points_y[ipt];
      point.z = points_z[ipt];


      const auto x = point.x - O.x;
      const auto y = point.y - O.y;
      const auto z = point.z - O.z;
      const auto rsq = x*x + y*y + z*z;

      // Evaluate radial part of bfn
      double radial_eval = 0.;
      double radial_eval_alpha = 0.;

      #pragma unroll 1
      for( uint32_t i = 0; i < nprim; ++i ) {
        const auto a = my_alpha[i];
        const auto e = my_coeff[i] * std::exp( - a * rsq );

        radial_eval += e;
        radial_eval_alpha += a * e;
      }

      radial_eval_alpha *= -2;

      // Common Subexpressions
      const auto x0 = x*x*x*x*x; 
      const auto x1 = radial_eval*y; 
      const auto x2 = x*x*x*x; 
      const auto x3 = radial_eval*z; 
      const auto x4 = x*x*x; 
      const auto x5 = y*y; 
      const auto x6 = x4*x5; 
      const auto x7 = z*z; 
      const auto x8 = x4*x7; 
      const auto x9 = x*x; 
      const auto x10 = y*y*y; 
      const auto x11 = x10*x9; 
      const auto x12 = x5*x9; 
      const auto x13 = x7*x9; 
      const auto x14 = z*z*z; 
      const auto x15 = x14*x9; 
      const auto x16 = radial_eval*x; 
      const auto x17 = y*y*y*y; 
      const auto x18 = x5*x7; 
      const auto x19 = z*z*z*z; 
      const auto x20 = y*y*y*y*y; 
      const auto x21 = x10*x7; 
      const auto x22 = x14*x5; 
      const auto x23 = z*z*z*z*z; 
      const auto x24 = 5.0*radial_eval; 
      const auto x25 = 4.0*radial_eval; 
      const auto x26 = radial_eval_alpha*x0 + x25*x4; 
      const auto x27 = radial_eval_alpha*x2*x5; 
      const auto x28 = 3.0*radial_eval; 
      const auto x29 = x12*x28; 
      const auto x30 = y*z; 
      const auto x31 = radial_eval_alpha*x2*x7; 
      const auto x32 = x13*x28; 
      const auto x33 = 2.0*x16; 
      const auto x34 = radial_eval_alpha*x10*x4; 
      const auto x35 = radial_eval_alpha*x6; 
      const auto x36 = radial_eval_alpha*x8; 
      const auto x37 = radial_eval_alpha*x14*x4; 
      const auto x38 = radial_eval*x17; 
      const auto x39 = radial_eval_alpha*x17*x9; 
      const auto x40 = radial_eval*x10; 
      const auto x41 = radial_eval_alpha*x11; 
      const auto x42 = radial_eval_alpha*x5*x7*x9; 
      const auto x43 = radial_eval*x14; 
      const auto x44 = radial_eval_alpha*x15; 
      const auto x45 = radial_eval*x19; 
      const auto x46 = radial_eval_alpha*x19*x9; 
      const auto x47 = radial_eval_alpha*x; 
      const auto x48 = x17*x47*z; 
      const auto x49 = x19*x47*y; 
      const auto x50 = radial_eval_alpha*y; 
      const auto x51 = radial_eval*x2; 
      const auto x52 = radial_eval_alpha*x2*x30; 
      const auto x53 = 2.0*x1; 
      const auto x54 = radial_eval*x4; 
      const auto x55 = radial_eval_alpha*x20 + x10*x25; 
      const auto x56 = radial_eval_alpha*x21; 
      const auto x57 = radial_eval_alpha*x22; 
      const auto x58 = radial_eval_alpha*x17*x7; 
      const auto x59 = x18*x28; 
      const auto x60 = radial_eval_alpha*x10*x14; 
      const auto x61 = radial_eval_alpha*x19*x5; 
      const auto x62 = radial_eval_alpha*z; 
      const auto x63 = 2.0*x3; 
      const auto x64 = radial_eval_alpha*x23 + x14*x25; 


      // Evaluate basis function
      basis_eval[ipt + 0*npts] = radial_eval*x0;
      basis_eval[ipt + 1*npts] = x1*x2;
      basis_eval[ipt + 2*npts] = x2*x3;
      basis_eval[ipt + 3*npts] = radial_eval*x6;
      basis_eval[ipt + 4*npts] = x1*x4*z;
      basis_eval[ipt + 5*npts] = radial_eval*x8;
      basis_eval[ipt + 6*npts] = radial_eval*x11;
      basis_eval[ipt + 7*npts] = x12*x3;
      basis_eval[ipt + 8*npts] = x1*x13;
      basis_eval[ipt + 9*npts] = radial_eval*x15;
      basis_eval[ipt + 10*npts] = x16*x17;
      basis_eval[ipt + 11*npts] = x*x10*x3;
      basis_eval[ipt + 12*npts] = x16*x18;
      basis_eval[ipt + 13*npts] = x*x1*x14;
      basis_eval[ipt + 14*npts] = x16*x19;
      basis_eval[ipt + 15*npts] = radial_eval*x20;
      basis_eval[ipt + 16*npts] = x17*x3;
      basis_eval[ipt + 17*npts] = radial_eval*x21;
      basis_eval[ipt + 18*npts] = radial_eval*x22;
      basis_eval[ipt + 19*npts] = x1*x19;
      basis_eval[ipt + 20*npts] = radial_eval*x23;


    
      // Evaluate first derivative of bfn wrt x
      basis_x_eval[ipt + 0*npts] = radial_eval_alpha*(x*x*x*x*x*x) + x2*x24;
      basis_x_eval[ipt + 1*npts] = x26*y;
      basis_x_eval[ipt + 2*npts] = x26*z;
      basis_x_eval[ipt + 3*npts] = x27 + x29;
      basis_x_eval[ipt + 4*npts] = x30*(radial_eval_alpha*x2 + x28*x9);
      basis_x_eval[ipt + 5*npts] = x31 + x32;
      basis_x_eval[ipt + 6*npts] = x10*x33 + x34;
      basis_x_eval[ipt + 7*npts] = z*(x33*x5 + x35);
      basis_x_eval[ipt + 8*npts] = y*(x33*x7 + x36);
      basis_x_eval[ipt + 9*npts] = x14*x33 + x37;
      basis_x_eval[ipt + 10*npts] = x38 + x39;
      basis_x_eval[ipt + 11*npts] = z*(x40 + x41);
      basis_x_eval[ipt + 12*npts] = radial_eval*x18 + x42;
      basis_x_eval[ipt + 13*npts] = y*(x43 + x44);
      basis_x_eval[ipt + 14*npts] = x45 + x46;
      basis_x_eval[ipt + 15*npts] = x20*x47;
      basis_x_eval[ipt + 16*npts] = x48;
      basis_x_eval[ipt + 17*npts] = x21*x47;
      basis_x_eval[ipt + 18*npts] = x22*x47;
      basis_x_eval[ipt + 19*npts] = x49;
      basis_x_eval[ipt + 20*npts] = x23*x47;

      // Evaluate first derivative of bfn wrt y
      basis_y_eval[ipt + 0*npts] = x0*x50;
      basis_y_eval[ipt + 1*npts] = x27 + x51;
      basis_y_eval[ipt + 2*npts] = x52;
      basis_y_eval[ipt + 3*npts] = x34 + x4*x53;
      basis_y_eval[ipt + 4*npts] = z*(x35 + x54);
      basis_y_eval[ipt + 5*npts] = x50*x8;
      basis_y_eval[ipt + 6*npts] = x29 + x39;
      basis_y_eval[ipt + 7*npts] = z*(x41 + x53*x9);
      basis_y_eval[ipt + 8*npts] = radial_eval*x13 + x42;
      basis_y_eval[ipt + 9*npts] = x15*x50;
      basis_y_eval[ipt + 10*npts] = x*x55;
      basis_y_eval[ipt + 11*npts] = x*z*(radial_eval_alpha*x17 + x28*x5);
      basis_y_eval[ipt + 12*npts] = x*(x53*x7 + x56);
      basis_y_eval[ipt + 13*npts] = x*(x43 + x57);
      basis_y_eval[ipt + 14*npts] = x49;
      basis_y_eval[ipt + 15*npts] = radial_eval_alpha*(y*y*y*y*y*y) + x17*x24;
      basis_y_eval[ipt + 16*npts] = x55*z;
      basis_y_eval[ipt + 17*npts] = x58 + x59;
      basis_y_eval[ipt + 18*npts] = x14*x53 + x60;
      basis_y_eval[ipt + 19*npts] = x45 + x61;
      basis_y_eval[ipt + 20*npts] = x23*x50;

      // Evaluate first derivative of bfn wrt z
      basis_z_eval[ipt + 0*npts] = x0*x62;
      basis_z_eval[ipt + 1*npts] = x52;
      basis_z_eval[ipt + 2*npts] = x31 + x51;
      basis_z_eval[ipt + 3*npts] = x6*x62;
      basis_z_eval[ipt + 4*npts] = y*(x36 + x54);
      basis_z_eval[ipt + 5*npts] = x37 + x4*x63;
      basis_z_eval[ipt + 6*npts] = x11*x62;
      basis_z_eval[ipt + 7*npts] = radial_eval*x12 + x42;
      basis_z_eval[ipt + 8*npts] = y*(x44 + x63*x9);
      basis_z_eval[ipt + 9*npts] = x32 + x46;
      basis_z_eval[ipt + 10*npts] = x48;
      basis_z_eval[ipt + 11*npts] = x*(x40 + x56);
      basis_z_eval[ipt + 12*npts] = x*(x5*x63 + x57);
      basis_z_eval[ipt + 13*npts] = x*y*(radial_eval_alpha*x19 + x28*x7);
      basis_z_eval[ipt + 14*npts] = x*x64;
      basis_z_eval[ipt + 15*npts] = x20*x62;
      basis_z_eval[ipt + 16*npts] = x38 + x58;
      basis_z_eval[ipt + 17*npts] = x10*x63 + x60;
      basis_z_eval[ipt + 18*npts] = x59 + x61;
      basis_z_eval[ipt + 19*npts] = x64*y;
      basis_z_eval[ipt + 20*npts] = radial_eval_alpha*(z*z*z*z*z*z) + x19*x24;







#if 0
      // Evaluate the angular part of bfn



      double ang_eval_0;
      double ang_eval_1;
      double ang_eval_2;
      double ang_eval_3;


      ang_eval_0 = radial_eval*x0;
      ang_eval_1 = x1*x2;
      ang_eval_2 = x2*x3;
      ang_eval_3 = radial_eval*x6;
      basis_eval[ipt + 0*npts] = ang_eval_0;
      basis_eval[ipt + 1*npts] = ang_eval_1;
      basis_eval[ipt + 2*npts] = ang_eval_2;
      basis_eval[ipt + 3*npts] = ang_eval_3;

      ang_eval_0 = x1*x4*z;
      ang_eval_1 = radial_eval*x8;
      ang_eval_2 = radial_eval*x11;
      ang_eval_3 = x12*x3;
      basis_eval[ipt + 4*npts] = ang_eval_0;
      basis_eval[ipt + 5*npts] = ang_eval_1;
      basis_eval[ipt + 6*npts] = ang_eval_2;
      basis_eval[ipt + 7*npts] = ang_eval_3;

      ang_eval_0 = x1*x13;
      ang_eval_1 = radial_eval*x15;
      ang_eval_2 = x16*x17;
      ang_eval_3 = x*x10*x3;
      basis_eval[ipt + 8*npts] = ang_eval_0;
      basis_eval[ipt + 9*npts] = ang_eval_1;
      basis_eval[ipt + 10*npts] = ang_eval_2;
      basis_eval[ipt + 11*npts] = ang_eval_3;

      ang_eval_0 = x16*x18;
      ang_eval_1 = x*x1*x14;
      ang_eval_2 = x16*x19;
      ang_eval_3 = radial_eval*x20;
      basis_eval[ipt + 12*npts] = ang_eval_0;
      basis_eval[ipt + 13*npts] = ang_eval_1;
      basis_eval[ipt + 14*npts] = ang_eval_2;
      basis_eval[ipt + 15*npts] = ang_eval_3;

      ang_eval_0 = x17*x3;
      ang_eval_1 = radial_eval*x21;
      ang_eval_2 = radial_eval*x22;
      ang_eval_3 = x1*x19;
      basis_eval[ipt + 16*npts] = ang_eval_0;
      basis_eval[ipt + 17*npts] = ang_eval_1;
      basis_eval[ipt + 18*npts] = ang_eval_2;
      basis_eval[ipt + 19*npts] = ang_eval_3;

      ang_eval_0 = radial_eval*x23;
      basis_eval[ipt + 20*npts] = ang_eval_0;


      double dang_eval_x_0, dang_eval_y_0, dang_eval_z_0;
      double dang_eval_x_1, dang_eval_y_1, dang_eval_z_1;
      double dang_eval_x_2, dang_eval_y_2, dang_eval_z_2;
      double dang_eval_x_3, dang_eval_y_3, dang_eval_z_3;

      dang_eval_x_0 = radial_eval_alpha*(x*x*x*x*x*x) + x2*x24;
      dang_eval_y_0 = x0*x50;
      dang_eval_z_0 = x0*x62;
      dang_eval_x_1 = x26*y;
      dang_eval_y_1 = x27 + x51;
      dang_eval_z_1 = x52;
      dang_eval_x_2 = x26*z;
      dang_eval_y_2 = x52;
      dang_eval_z_2 = x31 + x51;
      dang_eval_x_3 = x27 + x29;
      dang_eval_y_3 = x34 + x4*x53;
      dang_eval_z_3 = x6*x62;
      basis_x_eval[ipt + 0*npts] = dang_eval_x_0;
      basis_y_eval[ipt + 0*npts] = dang_eval_y_0;
      basis_z_eval[ipt + 0*npts] = dang_eval_z_0;
      basis_x_eval[ipt + 1*npts] = dang_eval_x_1;
      basis_y_eval[ipt + 1*npts] = dang_eval_y_1;
      basis_z_eval[ipt + 1*npts] = dang_eval_z_1;
      basis_x_eval[ipt + 2*npts] = dang_eval_x_2;
      basis_y_eval[ipt + 2*npts] = dang_eval_y_2;
      basis_z_eval[ipt + 2*npts] = dang_eval_z_2;
      basis_x_eval[ipt + 3*npts] = dang_eval_x_3;
      basis_y_eval[ipt + 3*npts] = dang_eval_y_3;
      basis_z_eval[ipt + 3*npts] = dang_eval_z_3;

      dang_eval_x_0 = x30*(radial_eval_alpha*x2 + x28*x9);
      dang_eval_y_0 = z*(x35 + x54);
      dang_eval_z_0 = y*(x36 + x54);
      dang_eval_x_1 = x31 + x32;
      dang_eval_y_1 = x50*x8;
      dang_eval_z_1 = x37 + x4*x63;
      dang_eval_x_2 = x10*x33 + x34;
      dang_eval_y_2 = x29 + x39;
      dang_eval_z_2 = x11*x62;
      dang_eval_x_3 = z*(x33*x5 + x35);
      dang_eval_y_3 = z*(x41 + x53*x9);
      dang_eval_z_3 = radial_eval*x12 + x42;
      basis_x_eval[ipt + 4*npts] = dang_eval_x_0;
      basis_y_eval[ipt + 4*npts] = dang_eval_y_0;
      basis_z_eval[ipt + 4*npts] = dang_eval_z_0;
      basis_x_eval[ipt + 5*npts] = dang_eval_x_1;
      basis_y_eval[ipt + 5*npts] = dang_eval_y_1;
      basis_z_eval[ipt + 5*npts] = dang_eval_z_1;
      basis_x_eval[ipt + 6*npts] = dang_eval_x_2;
      basis_y_eval[ipt + 6*npts] = dang_eval_y_2;
      basis_z_eval[ipt + 6*npts] = dang_eval_z_2;
      basis_x_eval[ipt + 7*npts] = dang_eval_x_3;
      basis_y_eval[ipt + 7*npts] = dang_eval_y_3;
      basis_z_eval[ipt + 7*npts] = dang_eval_z_3;

      dang_eval_x_0 = y*(x33*x7 + x36);
      dang_eval_y_0 = radial_eval*x13 + x42;
      dang_eval_z_0 = y*(x44 + x63*x9);
      dang_eval_x_1 = x14*x33 + x37;
      dang_eval_y_1 = x15*x50;
      dang_eval_z_1 = x32 + x46;
      dang_eval_x_2 = x38 + x39;
      dang_eval_y_2 = x*x55;
      dang_eval_z_2 = x48;
      dang_eval_x_3 = z*(x40 + x41);
      dang_eval_y_3 = x*z*(radial_eval_alpha*x17 + x28*x5);
      dang_eval_z_3 = x*(x40 + x56);
      basis_x_eval[ipt + 8*npts] = dang_eval_x_0;
      basis_y_eval[ipt + 8*npts] = dang_eval_y_0;
      basis_z_eval[ipt + 8*npts] = dang_eval_z_0;
      basis_x_eval[ipt + 9*npts] = dang_eval_x_1;
      basis_y_eval[ipt + 9*npts] = dang_eval_y_1;
      basis_z_eval[ipt + 9*npts] = dang_eval_z_1;
      basis_x_eval[ipt + 10*npts] = dang_eval_x_2;
      basis_y_eval[ipt + 10*npts] = dang_eval_y_2;
      basis_z_eval[ipt + 10*npts] = dang_eval_z_2;
      basis_x_eval[ipt + 11*npts] = dang_eval_x_3;
      basis_y_eval[ipt + 11*npts] = dang_eval_y_3;
      basis_z_eval[ipt + 11*npts] = dang_eval_z_3;

      dang_eval_x_0 = radial_eval*x18 + x42;
      dang_eval_y_0 = x*(x53*x7 + x56);
      dang_eval_z_0 = x*(x5*x63 + x57);
      dang_eval_x_1 = y*(x43 + x44);
      dang_eval_y_1 = x*(x43 + x57);
      dang_eval_z_1 = x*y*(radial_eval_alpha*x19 + x28*x7);
      dang_eval_x_2 = x45 + x46;
      dang_eval_y_2 = x49;
      dang_eval_z_2 = x*x64;
      dang_eval_x_3 = x20*x47;
      dang_eval_y_3 = radial_eval_alpha*(y*y*y*y*y*y) + x17*x24;
      dang_eval_z_3 = x20*x62;
      basis_x_eval[ipt + 12*npts] = dang_eval_x_0;
      basis_y_eval[ipt + 12*npts] = dang_eval_y_0;
      basis_z_eval[ipt + 12*npts] = dang_eval_z_0;
      basis_x_eval[ipt + 13*npts] = dang_eval_x_1;
      basis_y_eval[ipt + 13*npts] = dang_eval_y_1;
      basis_z_eval[ipt + 13*npts] = dang_eval_z_1;
      basis_x_eval[ipt + 14*npts] = dang_eval_x_2;
      basis_y_eval[ipt + 14*npts] = dang_eval_y_2;
      basis_z_eval[ipt + 14*npts] = dang_eval_z_2;
      basis_x_eval[ipt + 15*npts] = dang_eval_x_3;
      basis_y_eval[ipt + 15*npts] = dang_eval_y_3;
      basis_z_eval[ipt + 15*npts] = dang_eval_z_3;

      dang_eval_x_0 = x48;
      dang_eval_y_0 = x55*z;
      dang_eval_z_0 = x38 + x58;
      dang_eval_x_1 = x21*x47;
      dang_eval_y_1 = x58 + x59;
      dang_eval_z_1 = x10*x63 + x60;
      dang_eval_x_2 = x22*x47;
      dang_eval_y_2 = x14*x53 + x60;
      dang_eval_z_2 = x59 + x61;
      dang_eval_x_3 = x49;
      dang_eval_y_3 = x45 + x61;
      dang_eval_z_3 = x64*y;
      basis_x_eval[ipt + 16*npts] = dang_eval_x_0;
      basis_y_eval[ipt + 16*npts] = dang_eval_y_0;
      basis_z_eval[ipt + 16*npts] = dang_eval_z_0;
      basis_x_eval[ipt + 17*npts] = dang_eval_x_1;
      basis_y_eval[ipt + 17*npts] = dang_eval_y_1;
      basis_z_eval[ipt + 17*npts] = dang_eval_z_1;
      basis_x_eval[ipt + 18*npts] = dang_eval_x_2;
      basis_y_eval[ipt + 18*npts] = dang_eval_y_2;
      basis_z_eval[ipt + 18*npts] = dang_eval_z_2;
      basis_x_eval[ipt + 19*npts] = dang_eval_x_3;
      basis_y_eval[ipt + 19*npts] = dang_eval_y_3;
      basis_z_eval[ipt + 19*npts] = dang_eval_z_3;

      dang_eval_x_0 = x23*x47;
      dang_eval_y_0 = x23*x50;
      dang_eval_z_0 = radial_eval_alpha*(z*z*z*z*z*z) + x19*x24;
      basis_x_eval[ipt + 20*npts] = dang_eval_x_0;
      basis_y_eval[ipt + 20*npts] = dang_eval_y_0;
      basis_z_eval[ipt + 20*npts] = dang_eval_z_0;

#endif
    } // Loop over points within task
  } // Loop over tasks
        
  } // Loop over shells
} // end kernel

} // namespace GauXC
