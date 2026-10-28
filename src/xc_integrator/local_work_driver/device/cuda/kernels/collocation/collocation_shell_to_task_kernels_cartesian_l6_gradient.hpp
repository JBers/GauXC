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
#pragma once
#include "collocation_device_constants.hpp"
#include "device/xc_device_task.hpp"
#include "device_specific/cuda_device_constants.hpp"
#include "device/common/shell_to_task.hpp"
#include <cassert>

namespace GauXC {


__global__ __launch_bounds__(128,2) void collocation_device_shell_to_task_kernel_cartesian_gradient_6(
  uint32_t                        nshell,
  ShellToTaskDevice* __restrict__ shell_to_task,
  XCDeviceTask*      __restrict__ device_tasks
) {


  __shared__ double alpha[128/cuda::warp_size)][detail::shell_nprim_max + 1]; 
  __shared__ double coeff[128/cuda::warp_size)][detail::shell_nprim_max + 1];
  double* my_alpha = alpha[threadIdx.x/cuda::warp_size];
  double* my_coeff = coeff[threadIdx.x/cuda::warp_size];

  for( auto ish = blockIdx.z; ish < nshell; ish += gridDim.z ) {
  const uint32_t ntasks      = shell_to_task[ish].ntask;
  const auto shell           = shell_to_task[ish].shell_device;
  const auto task_idx        = shell_to_task[ish].task_idx_device;
  const auto task_shell_offs = shell_to_task[ish].task_shell_offs_device;


  // Load Shell Data into registers / SM
  const uint32_t nprim = shell->nprim();
  const double3 O  = *reinterpret_cast<const double3*>(shell->O_data());

  const int global_warp_id = (threadIdx.x + blockIdx.x*blockDim.x) / cuda::warp_size;
  const int nwarp_global   = max((blockDim.x*gridDim.x) / cuda::warp_size,1);

  // Read in coeffs/exps into SM on first warp
  {
    auto* coeff_gm = shell->coeff_data();
    auto* alpha_gm = shell->alpha_data();
    static_assert( detail::shell_nprim_max <= cuda::warp_size );
    const int warp_rank = threadIdx.x % cuda::warp_size;
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
    for( int ipt = threadIdx.x % cuda::warp_size; ipt < npts; ipt += cuda::warp_size ) {
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
      const auto x0 = x*x*x*x*x*x; 
      const auto x1 = radial_eval*y; 
      const auto x2 = x*x*x*x*x; 
      const auto x3 = radial_eval*z; 
      const auto x4 = x*x*x*x; 
      const auto x5 = y*y; 
      const auto x6 = x4*x5; 
      const auto x7 = z*z; 
      const auto x8 = x4*x7; 
      const auto x9 = x*x*x; 
      const auto x10 = y*y*y; 
      const auto x11 = x10*x9; 
      const auto x12 = x5*x9; 
      const auto x13 = x7*x9; 
      const auto x14 = z*z*z; 
      const auto x15 = x14*x9; 
      const auto x16 = x*x; 
      const auto x17 = y*y*y*y; 
      const auto x18 = x16*x17; 
      const auto x19 = x10*x16; 
      const auto x20 = x14*x16; 
      const auto x21 = z*z*z*z; 
      const auto x22 = x16*x21; 
      const auto x23 = radial_eval*x; 
      const auto x24 = y*y*y*y*y; 
      const auto x25 = x10*x7; 
      const auto x26 = x14*x5; 
      const auto x27 = z*z*z*z*z; 
      const auto x28 = y*y*y*y*y*y; 
      const auto x29 = x17*x7; 
      const auto x30 = x10*x14; 
      const auto x31 = x21*x5; 
      const auto x32 = z*z*z*z*z*z; 
      const auto x33 = 6.0*radial_eval; 
      const auto x34 = 5.0*radial_eval; 
      const auto x35 = radial_eval_alpha*x0 + x34*x4; 
      const auto x36 = radial_eval_alpha*x2*x5; 
      const auto x37 = 4.0*radial_eval; 
      const auto x38 = y*z; 
      const auto x39 = radial_eval_alpha*x2*x7; 
      const auto x40 = radial_eval_alpha*x10*x4; 
      const auto x41 = 3.0*radial_eval; 
      const auto x42 = radial_eval_alpha*x6; 
      const auto x43 = x16*x5; 
      const auto x44 = x41*x43; 
      const auto x45 = radial_eval_alpha*x8; 
      const auto x46 = x16*x7; 
      const auto x47 = x41*x46; 
      const auto x48 = radial_eval_alpha*x14*x4; 
      const auto x49 = 2.0*x23; 
      const auto x50 = radial_eval_alpha*x17*x9; 
      const auto x51 = radial_eval_alpha*x11; 
      const auto x52 = x5*x7; 
      const auto x53 = radial_eval_alpha*x5*x7*x9; 
      const auto x54 = radial_eval_alpha*x15; 
      const auto x55 = radial_eval_alpha*x21*x9; 
      const auto x56 = radial_eval*x24; 
      const auto x57 = radial_eval_alpha*x16*x24; 
      const auto x58 = radial_eval*x17; 
      const auto x59 = radial_eval_alpha*x18; 
      const auto x60 = radial_eval_alpha*x10*x16*x7; 
      const auto x61 = radial_eval_alpha*x14*x16*x5; 
      const auto x62 = radial_eval*x21; 
      const auto x63 = radial_eval_alpha*x22; 
      const auto x64 = radial_eval*x27; 
      const auto x65 = radial_eval_alpha*x16*x27; 
      const auto x66 = radial_eval_alpha*x; 
      const auto x67 = x24*x66*z; 
      const auto x68 = x27*x66*y; 
      const auto x69 = radial_eval_alpha*y; 
      const auto x70 = radial_eval*x2; 
      const auto x71 = radial_eval_alpha*x2*x38; 
      const auto x72 = 2.0*x1; 
      const auto x73 = radial_eval*x4; 
      const auto x74 = radial_eval_alpha*x28 + x17*x34; 
      const auto x75 = radial_eval_alpha*x29; 
      const auto x76 = x41*x52; 
      const auto x77 = radial_eval_alpha*x30; 
      const auto x78 = radial_eval_alpha*x31; 
      const auto x79 = radial_eval_alpha*x24*x7; 
      const auto x80 = radial_eval_alpha*x14*x17; 
      const auto x81 = radial_eval_alpha*x10*x21; 
      const auto x82 = radial_eval_alpha*x27*x5; 
      const auto x83 = radial_eval_alpha*z; 
      const auto x84 = 2.0*x3; 
      const auto x85 = radial_eval_alpha*x32 + x21*x34; 


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
      basis_eval[ipt + 10*npts] = radial_eval*x18;
      basis_eval[ipt + 11*npts] = x19*x3;
      basis_eval[ipt + 12*npts] = radial_eval*x16*x5*x7;
      basis_eval[ipt + 13*npts] = x1*x20;
      basis_eval[ipt + 14*npts] = radial_eval*x22;
      basis_eval[ipt + 15*npts] = x23*x24;
      basis_eval[ipt + 16*npts] = x*x17*x3;
      basis_eval[ipt + 17*npts] = x23*x25;
      basis_eval[ipt + 18*npts] = x23*x26;
      basis_eval[ipt + 19*npts] = x*x1*x21;
      basis_eval[ipt + 20*npts] = x23*x27;
      basis_eval[ipt + 21*npts] = radial_eval*x28;
      basis_eval[ipt + 22*npts] = x24*x3;
      basis_eval[ipt + 23*npts] = radial_eval*x29;
      basis_eval[ipt + 24*npts] = radial_eval*x30;
      basis_eval[ipt + 25*npts] = radial_eval*x31;
      basis_eval[ipt + 26*npts] = x1*x27;
      basis_eval[ipt + 27*npts] = radial_eval*x32;


    
      // Evaluate first derivative of bfn wrt x
      basis_x_eval[ipt + 0*npts] = radial_eval_alpha*(x*x*x*x*x*x*x) + x2*x33;
      basis_x_eval[ipt + 1*npts] = x35*y;
      basis_x_eval[ipt + 2*npts] = x35*z;
      basis_x_eval[ipt + 3*npts] = x12*x37 + x36;
      basis_x_eval[ipt + 4*npts] = x38*(radial_eval_alpha*x2 + x37*x9);
      basis_x_eval[ipt + 5*npts] = x13*x37 + x39;
      basis_x_eval[ipt + 6*npts] = x19*x41 + x40;
      basis_x_eval[ipt + 7*npts] = z*(x42 + x44);
      basis_x_eval[ipt + 8*npts] = y*(x45 + x47);
      basis_x_eval[ipt + 9*npts] = x20*x41 + x48;
      basis_x_eval[ipt + 10*npts] = x17*x49 + x50;
      basis_x_eval[ipt + 11*npts] = z*(x10*x49 + x51);
      basis_x_eval[ipt + 12*npts] = x49*x52 + x53;
      basis_x_eval[ipt + 13*npts] = y*(x14*x49 + x54);
      basis_x_eval[ipt + 14*npts] = x21*x49 + x55;
      basis_x_eval[ipt + 15*npts] = x56 + x57;
      basis_x_eval[ipt + 16*npts] = z*(x58 + x59);
      basis_x_eval[ipt + 17*npts] = radial_eval*x25 + x60;
      basis_x_eval[ipt + 18*npts] = radial_eval*x26 + x61;
      basis_x_eval[ipt + 19*npts] = y*(x62 + x63);
      basis_x_eval[ipt + 20*npts] = x64 + x65;
      basis_x_eval[ipt + 21*npts] = x28*x66;
      basis_x_eval[ipt + 22*npts] = x67;
      basis_x_eval[ipt + 23*npts] = x29*x66;
      basis_x_eval[ipt + 24*npts] = x30*x66;
      basis_x_eval[ipt + 25*npts] = x31*x66;
      basis_x_eval[ipt + 26*npts] = x68;
      basis_x_eval[ipt + 27*npts] = x32*x66;

      // Evaluate first derivative of bfn wrt y
      basis_y_eval[ipt + 0*npts] = x0*x69;
      basis_y_eval[ipt + 1*npts] = x36 + x70;
      basis_y_eval[ipt + 2*npts] = x71;
      basis_y_eval[ipt + 3*npts] = x4*x72 + x40;
      basis_y_eval[ipt + 4*npts] = z*(x42 + x73);
      basis_y_eval[ipt + 5*npts] = x69*x8;
      basis_y_eval[ipt + 6*npts] = x12*x41 + x50;
      basis_y_eval[ipt + 7*npts] = z*(x51 + x72*x9);
      basis_y_eval[ipt + 8*npts] = radial_eval*x13 + x53;
      basis_y_eval[ipt + 9*npts] = x15*x69;
      basis_y_eval[ipt + 10*npts] = x19*x37 + x57;
      basis_y_eval[ipt + 11*npts] = z*(x44 + x59);
      basis_y_eval[ipt + 12*npts] = x46*x72 + x60;
      basis_y_eval[ipt + 13*npts] = radial_eval*x20 + x61;
      basis_y_eval[ipt + 14*npts] = x22*x69;
      basis_y_eval[ipt + 15*npts] = x*x74;
      basis_y_eval[ipt + 16*npts] = x*z*(radial_eval_alpha*x24 + x10*x37);
      basis_y_eval[ipt + 17*npts] = x*(x75 + x76);
      basis_y_eval[ipt + 18*npts] = x*(x14*x72 + x77);
      basis_y_eval[ipt + 19*npts] = x*(x62 + x78);
      basis_y_eval[ipt + 20*npts] = x68;
      basis_y_eval[ipt + 21*npts] = radial_eval_alpha*(y*y*y*y*y*y*y) + x24*x33;
      basis_y_eval[ipt + 22*npts] = x74*z;
      basis_y_eval[ipt + 23*npts] = x25*x37 + x79;
      basis_y_eval[ipt + 24*npts] = x26*x41 + x80;
      basis_y_eval[ipt + 25*npts] = x21*x72 + x81;
      basis_y_eval[ipt + 26*npts] = x64 + x82;
      basis_y_eval[ipt + 27*npts] = x32*x69;

      // Evaluate first derivative of bfn wrt z
      basis_z_eval[ipt + 0*npts] = x0*x83;
      basis_z_eval[ipt + 1*npts] = x71;
      basis_z_eval[ipt + 2*npts] = x39 + x70;
      basis_z_eval[ipt + 3*npts] = x6*x83;
      basis_z_eval[ipt + 4*npts] = y*(x45 + x73);
      basis_z_eval[ipt + 5*npts] = x4*x84 + x48;
      basis_z_eval[ipt + 6*npts] = x11*x83;
      basis_z_eval[ipt + 7*npts] = radial_eval*x12 + x53;
      basis_z_eval[ipt + 8*npts] = y*(x54 + x84*x9);
      basis_z_eval[ipt + 9*npts] = x13*x41 + x55;
      basis_z_eval[ipt + 10*npts] = x18*x83;
      basis_z_eval[ipt + 11*npts] = radial_eval*x19 + x60;
      basis_z_eval[ipt + 12*npts] = x43*x84 + x61;
      basis_z_eval[ipt + 13*npts] = y*(x47 + x63);
      basis_z_eval[ipt + 14*npts] = x20*x37 + x65;
      basis_z_eval[ipt + 15*npts] = x67;
      basis_z_eval[ipt + 16*npts] = x*(x58 + x75);
      basis_z_eval[ipt + 17*npts] = x*(x10*x84 + x77);
      basis_z_eval[ipt + 18*npts] = x*(x76 + x78);
      basis_z_eval[ipt + 19*npts] = x*y*(radial_eval_alpha*x27 + x14*x37);
      basis_z_eval[ipt + 20*npts] = x*x85;
      basis_z_eval[ipt + 21*npts] = x28*x83;
      basis_z_eval[ipt + 22*npts] = x56 + x79;
      basis_z_eval[ipt + 23*npts] = x17*x84 + x80;
      basis_z_eval[ipt + 24*npts] = x25*x41 + x81;
      basis_z_eval[ipt + 25*npts] = x26*x37 + x82;
      basis_z_eval[ipt + 26*npts] = x85*y;
      basis_z_eval[ipt + 27*npts] = radial_eval_alpha*(z*z*z*z*z*z*z) + x27*x33;







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
      ang_eval_2 = radial_eval*x18;
      ang_eval_3 = x19*x3;
      basis_eval[ipt + 8*npts] = ang_eval_0;
      basis_eval[ipt + 9*npts] = ang_eval_1;
      basis_eval[ipt + 10*npts] = ang_eval_2;
      basis_eval[ipt + 11*npts] = ang_eval_3;

      ang_eval_0 = radial_eval*x16*x5*x7;
      ang_eval_1 = x1*x20;
      ang_eval_2 = radial_eval*x22;
      ang_eval_3 = x23*x24;
      basis_eval[ipt + 12*npts] = ang_eval_0;
      basis_eval[ipt + 13*npts] = ang_eval_1;
      basis_eval[ipt + 14*npts] = ang_eval_2;
      basis_eval[ipt + 15*npts] = ang_eval_3;

      ang_eval_0 = x*x17*x3;
      ang_eval_1 = x23*x25;
      ang_eval_2 = x23*x26;
      ang_eval_3 = x*x1*x21;
      basis_eval[ipt + 16*npts] = ang_eval_0;
      basis_eval[ipt + 17*npts] = ang_eval_1;
      basis_eval[ipt + 18*npts] = ang_eval_2;
      basis_eval[ipt + 19*npts] = ang_eval_3;

      ang_eval_0 = x23*x27;
      ang_eval_1 = radial_eval*x28;
      ang_eval_2 = x24*x3;
      ang_eval_3 = radial_eval*x29;
      basis_eval[ipt + 20*npts] = ang_eval_0;
      basis_eval[ipt + 21*npts] = ang_eval_1;
      basis_eval[ipt + 22*npts] = ang_eval_2;
      basis_eval[ipt + 23*npts] = ang_eval_3;

      ang_eval_0 = radial_eval*x30;
      ang_eval_1 = radial_eval*x31;
      ang_eval_2 = x1*x27;
      ang_eval_3 = radial_eval*x32;
      basis_eval[ipt + 24*npts] = ang_eval_0;
      basis_eval[ipt + 25*npts] = ang_eval_1;
      basis_eval[ipt + 26*npts] = ang_eval_2;
      basis_eval[ipt + 27*npts] = ang_eval_3;


      double dang_eval_x_0, dang_eval_y_0, dang_eval_z_0;
      double dang_eval_x_1, dang_eval_y_1, dang_eval_z_1;
      double dang_eval_x_2, dang_eval_y_2, dang_eval_z_2;
      double dang_eval_x_3, dang_eval_y_3, dang_eval_z_3;

      dang_eval_x_0 = radial_eval_alpha*(x*x*x*x*x*x*x) + x2*x33;
      dang_eval_y_0 = x0*x69;
      dang_eval_z_0 = x0*x83;
      dang_eval_x_1 = x35*y;
      dang_eval_y_1 = x36 + x70;
      dang_eval_z_1 = x71;
      dang_eval_x_2 = x35*z;
      dang_eval_y_2 = x71;
      dang_eval_z_2 = x39 + x70;
      dang_eval_x_3 = x12*x37 + x36;
      dang_eval_y_3 = x4*x72 + x40;
      dang_eval_z_3 = x6*x83;
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

      dang_eval_x_0 = x38*(radial_eval_alpha*x2 + x37*x9);
      dang_eval_y_0 = z*(x42 + x73);
      dang_eval_z_0 = y*(x45 + x73);
      dang_eval_x_1 = x13*x37 + x39;
      dang_eval_y_1 = x69*x8;
      dang_eval_z_1 = x4*x84 + x48;
      dang_eval_x_2 = x19*x41 + x40;
      dang_eval_y_2 = x12*x41 + x50;
      dang_eval_z_2 = x11*x83;
      dang_eval_x_3 = z*(x42 + x44);
      dang_eval_y_3 = z*(x51 + x72*x9);
      dang_eval_z_3 = radial_eval*x12 + x53;
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

      dang_eval_x_0 = y*(x45 + x47);
      dang_eval_y_0 = radial_eval*x13 + x53;
      dang_eval_z_0 = y*(x54 + x84*x9);
      dang_eval_x_1 = x20*x41 + x48;
      dang_eval_y_1 = x15*x69;
      dang_eval_z_1 = x13*x41 + x55;
      dang_eval_x_2 = x17*x49 + x50;
      dang_eval_y_2 = x19*x37 + x57;
      dang_eval_z_2 = x18*x83;
      dang_eval_x_3 = z*(x10*x49 + x51);
      dang_eval_y_3 = z*(x44 + x59);
      dang_eval_z_3 = radial_eval*x19 + x60;
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

      dang_eval_x_0 = x49*x52 + x53;
      dang_eval_y_0 = x46*x72 + x60;
      dang_eval_z_0 = x43*x84 + x61;
      dang_eval_x_1 = y*(x14*x49 + x54);
      dang_eval_y_1 = radial_eval*x20 + x61;
      dang_eval_z_1 = y*(x47 + x63);
      dang_eval_x_2 = x21*x49 + x55;
      dang_eval_y_2 = x22*x69;
      dang_eval_z_2 = x20*x37 + x65;
      dang_eval_x_3 = x56 + x57;
      dang_eval_y_3 = x*x74;
      dang_eval_z_3 = x67;
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

      dang_eval_x_0 = z*(x58 + x59);
      dang_eval_y_0 = x*z*(radial_eval_alpha*x24 + x10*x37);
      dang_eval_z_0 = x*(x58 + x75);
      dang_eval_x_1 = radial_eval*x25 + x60;
      dang_eval_y_1 = x*(x75 + x76);
      dang_eval_z_1 = x*(x10*x84 + x77);
      dang_eval_x_2 = radial_eval*x26 + x61;
      dang_eval_y_2 = x*(x14*x72 + x77);
      dang_eval_z_2 = x*(x76 + x78);
      dang_eval_x_3 = y*(x62 + x63);
      dang_eval_y_3 = x*(x62 + x78);
      dang_eval_z_3 = x*y*(radial_eval_alpha*x27 + x14*x37);
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

      dang_eval_x_0 = x64 + x65;
      dang_eval_y_0 = x68;
      dang_eval_z_0 = x*x85;
      dang_eval_x_1 = x28*x66;
      dang_eval_y_1 = radial_eval_alpha*(y*y*y*y*y*y*y) + x24*x33;
      dang_eval_z_1 = x28*x83;
      dang_eval_x_2 = x67;
      dang_eval_y_2 = x74*z;
      dang_eval_z_2 = x56 + x79;
      dang_eval_x_3 = x29*x66;
      dang_eval_y_3 = x25*x37 + x79;
      dang_eval_z_3 = x17*x84 + x80;
      basis_x_eval[ipt + 20*npts] = dang_eval_x_0;
      basis_y_eval[ipt + 20*npts] = dang_eval_y_0;
      basis_z_eval[ipt + 20*npts] = dang_eval_z_0;
      basis_x_eval[ipt + 21*npts] = dang_eval_x_1;
      basis_y_eval[ipt + 21*npts] = dang_eval_y_1;
      basis_z_eval[ipt + 21*npts] = dang_eval_z_1;
      basis_x_eval[ipt + 22*npts] = dang_eval_x_2;
      basis_y_eval[ipt + 22*npts] = dang_eval_y_2;
      basis_z_eval[ipt + 22*npts] = dang_eval_z_2;
      basis_x_eval[ipt + 23*npts] = dang_eval_x_3;
      basis_y_eval[ipt + 23*npts] = dang_eval_y_3;
      basis_z_eval[ipt + 23*npts] = dang_eval_z_3;

      dang_eval_x_0 = x30*x66;
      dang_eval_y_0 = x26*x41 + x80;
      dang_eval_z_0 = x25*x41 + x81;
      dang_eval_x_1 = x31*x66;
      dang_eval_y_1 = x21*x72 + x81;
      dang_eval_z_1 = x26*x37 + x82;
      dang_eval_x_2 = x68;
      dang_eval_y_2 = x64 + x82;
      dang_eval_z_2 = x85*y;
      dang_eval_x_3 = x32*x66;
      dang_eval_y_3 = x32*x69;
      dang_eval_z_3 = radial_eval_alpha*(z*z*z*z*z*z*z) + x27*x33;
      basis_x_eval[ipt + 24*npts] = dang_eval_x_0;
      basis_y_eval[ipt + 24*npts] = dang_eval_y_0;
      basis_z_eval[ipt + 24*npts] = dang_eval_z_0;
      basis_x_eval[ipt + 25*npts] = dang_eval_x_1;
      basis_y_eval[ipt + 25*npts] = dang_eval_y_1;
      basis_z_eval[ipt + 25*npts] = dang_eval_z_1;
      basis_x_eval[ipt + 26*npts] = dang_eval_x_2;
      basis_y_eval[ipt + 26*npts] = dang_eval_y_2;
      basis_z_eval[ipt + 26*npts] = dang_eval_z_2;
      basis_x_eval[ipt + 27*npts] = dang_eval_x_3;
      basis_y_eval[ipt + 27*npts] = dang_eval_y_3;
      basis_z_eval[ipt + 27*npts] = dang_eval_z_3;

#endif
    } // Loop over points within task
  } // Loop over tasks
        
  } // Loop over shells
} // end kernel

} // namespace GauXC
