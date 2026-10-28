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


__global__ __launch_bounds__(128,2) void collocation_device_shell_to_task_kernel_spherical_gradient_6(
  uint32_t                        nshell,
  ShellToTaskDevice* __restrict__ shell_to_task,
  XCDeviceTask*      __restrict__ device_tasks
) {


  __shared__ double alpha[128/hip::warp_size)][detail::shell_nprim_max + 1]; 
  __shared__ double coeff[128/hip::warp_size)][detail::shell_nprim_max + 1];
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
      const auto x0 = sqrt_462*x; 
      const auto x1 = 0.0625*radial_eval; 
      const auto x2 = x1*y; 
      const auto x3 = x*x*x*x; 
      const auto x4 = 3.0*x3; 
      const auto x5 = y*y*y*y; 
      const auto x6 = 3.0*x5; 
      const auto x7 = x*x; 
      const auto x8 = y*y; 
      const auto x9 = x7*x8; 
      const auto x10 = 10.0*x9; 
      const auto x11 = -x10; 
      const auto x12 = x11 + x4 + x6; 
      const auto x13 = 0.1875*sqrt_154; 
      const auto x14 = x13*z; 
      const auto x15 = x14*y; 
      const auto x16 = 5.0*x3; 
      const auto x17 = x11 + x16 + x5; 
      const auto x18 = 0.75*sqrt_7; 
      const auto x19 = x18*y; 
      const auto x20 = x*x19; 
      const auto x21 = z*z; 
      const auto x22 = x21; 
      const auto x23 = x22*x7; 
      const auto x24 = x22*x8; 
      const auto x25 = 10.0*x23 - 10.0*x24 - x3 + x5; 
      const auto x26 = sqrt_210*x2; 
      const auto x27 = 9.0*x3; 
      const auto x28 = 6.0*x9; 
      const auto x29 = 8.0*x24; 
      const auto x30 = 24.0*x22*x7 - x27 - x28 - x29 + 3.0*x5; 
      const auto x31 = z*z*z*z; 
      const auto x32 = x3 + x5; 
      const auto x33 = -16.0*x23 - 16.0*x24 + 16.0*x31 + x32 + 2.0*x9; 
      const auto x34 = 0.125*sqrt_21; 
      const auto x35 = x34*z; 
      const auto x36 = x35*y; 
      const auto x37 = 5.0*x5; 
      const auto x38 = x10 + x16 - 20.0*x23 - 20.0*x24 + 8.0*x31 + x37; 
      const auto x39 = x*x*x*x*x*x; 
      const auto x40 = y*y*y*y*y*y; 
      const auto x41 = x5*x7; 
      const auto x42 = 15.0*x41; 
      const auto x43 = x31*x7; 
      const auto x44 = x3*x8; 
      const auto x45 = 15.0*x44; 
      const auto x46 = x31*x8; 
      const auto x47 = 90.0*x22*x3 + 90.0*x22*x5 + 180.0*x22*x7*x8 - 5.0*x39 - 5.0*x40 - x42 - 120.0*x43 - x45 - 120.0*x46 + 16.0*(z*z*z*z*z*z); 
      const auto x48 = x*x35; 
      const auto x49 = 0.03125*sqrt_210; 
      const auto x50 = x22*x3; 
      const auto x51 = x22*x5; 
      const auto x52 = -x40; 
      const auto x53 = x39 + x52; 
      const auto x54 = -x41 + 16.0*x43 + x44 - 16.0*x46 - 16.0*x50 + 16.0*x51 + x53; 
      const auto x55 = 8.0*x23 - 24.0*x24 + x28 - x4 + 9.0*x5; 
      const auto x56 = 0.1875*sqrt_7; 
      const auto x57 = -60.0*x22*x7*x8 - x39 + 5.0*x41 + 5.0*x44 + 10.0*x50 + 10.0*x51 + x52; 
      const auto x58 = x*x14; 
      const auto x59 = x11 + x3 + x37; 
      const auto x60 = 0.03125*sqrt_462; 
      const auto x61 = x42 - x45 + x53; 
      const auto x62 = 0.0625*y; 
      const auto x63 = sqrt_462*x62; 
      const auto x64 = 4.0*x; 
      const auto x65 = x*x*x; 
      const auto x66 = 3.0*x65; 
      const auto x67 = 5.0*x; 
      const auto x68 = -x67*x8; 
      const auto x69 = 20.0*radial_eval; 
      const auto x70 = x*x8; 
      const auto x71 = radial_eval_alpha*x; 
      const auto x72 = sqrt_210*x62; 
      const auto x73 = 12.0*radial_eval; 
      const auto x74 = 8.0*x; 
      const auto x75 = x65 + x70; 
      const auto x76 = -2.0*x*x22 + x75; 
      const auto x77 = 30.0*radial_eval; 
      const auto x78 = x*x*x*x*x; 
      const auto x79 = x*x5; 
      const auto x80 = x65*x8; 
      const auto x81 = 2.0*x80; 
      const auto x82 = 2.0*radial_eval; 
      const auto x83 = 3.0*x78; 
      const auto x84 = x22*x65; 
      const auto x85 = 0.0625*sqrt_210; 
      const auto x86 = x85*z; 
      const auto x87 = x5*x67; 
      const auto x88 = 10.0*x80; 
      const auto x89 = 6.0*radial_eval; 
      const auto x90 = 4.0*y; 
      const auto x91 = y*y*y; 
      const auto x92 = 3.0*x91; 
      const auto x93 = 5.0*y; 
      const auto x94 = -x7*x93; 
      const auto x95 = x22*x90; 
      const auto x96 = -x28; 
      const auto x97 = 24.0*x23 - x27 - x29 + x6 + x96; 
      const auto x98 = x*x85; 
      const auto x99 = 8.0*y; 
      const auto x100 = x7*y; 
      const auto x101 = x100 + x91; 
      const auto x102 = x101 - 2.0*x22*y; 
      const auto x103 = y*y*y*y*y; 
      const auto x104 = x3*y; 
      const auto x105 = x7*x91; 
      const auto x106 = 2.0*x105; 
      const auto x107 = radial_eval_alpha*y; 
      const auto x108 = 3.0*x103; 
      const auto x109 = x3*x93; 
      const auto x110 = -x8; 
      const auto x111 = 16.0*x21; 
      const auto x112 = 32.0*radial_eval; 
      const auto x113 = z*z*z; 
      const auto x114 = radial_eval_alpha*z; 
      const auto x115 = 5.0*z; 
      const auto x116 = radial_eval*(x38 - 8.0*z*(-4.0*x113 + x115*x7 + x115*x8)) + radial_eval_alpha*x22*x38; 
      const auto x117 = 15.0*z; 
      const auto x118 = x113*x7; 
      const auto x119 = x113*x8; 


      // Evaluate basis function
      basis_eval[ipt + 0*npts] = x0*x12*x2;
      basis_eval[ipt + 1*npts] = radial_eval*x15*x17;
      basis_eval[ipt + 2*npts] = radial_eval*x20*x25;
      basis_eval[ipt + 3*npts] = x26*x30*z;
      basis_eval[ipt + 4*npts] = x*x26*x33;
      basis_eval[ipt + 5*npts] = radial_eval*x36*x38;
      basis_eval[ipt + 6*npts] = x1*x47;
      basis_eval[ipt + 7*npts] = radial_eval*x38*x48;
      basis_eval[ipt + 8*npts] = radial_eval*x49*x54;
      basis_eval[ipt + 9*npts] = sqrt_210*x*x1*x55*z;
      basis_eval[ipt + 10*npts] = radial_eval*x56*x57;
      basis_eval[ipt + 11*npts] = radial_eval*x58*x59;
      basis_eval[ipt + 12*npts] = radial_eval*x60*x61;


    
      // Evaluate first derivative of bfn wrt x
      basis_x_eval[ipt + 0*npts] = x63*(radial_eval*(x12 + x64*(x66 + x68)) + radial_eval_alpha*x12*x7);
      basis_x_eval[ipt + 1*npts] = x15*(x17*x71 + x69*(x65 - x70));
      basis_x_eval[ipt + 2*npts] = x19*(radial_eval*(x25 + x64*(x22*x67 - x65)) + radial_eval_alpha*x25*x7);
      basis_x_eval[ipt + 3*npts] = x72*z*(x30*x71 - x73*(-4.0*x*x22 + x66 + x70));
      basis_x_eval[ipt + 4*npts] = x72*(radial_eval*(x33 - x64*(x22*x74 - x75)) + radial_eval_alpha*x33*x7);
      basis_x_eval[ipt + 5*npts] = x36*(x38*x71 + x69*x76);
      basis_x_eval[ipt + 6*npts] = 0.0625*x47*x71 - 0.0625*x77*(-12.0*x*x22*x8 - 12.0*x22*x65 + x31*x74 + x78 + x79 + x81);
      basis_x_eval[ipt + 7*npts] = x35*(radial_eval*(20.0*x*x76 + x38) + radial_eval_alpha*x38*x7);
      basis_x_eval[ipt + 8*npts] = x49*(x54*x71 + x82*(16.0*x*x31 - x79 + x81 + x83 - 32.0*x84));
      basis_x_eval[ipt + 9*npts] = x86*(radial_eval*(x55 + x64*(3.0*x*x8 + x22*x64 - x66)) + radial_eval_alpha*x55*x7);
      basis_x_eval[ipt + 10*npts] = x56*(x57*x71 + x82*(-60.0*x*x24 - x83 + 20.0*x84 + x87 + x88));
      basis_x_eval[ipt + 11*npts] = x14*(radial_eval*(x59 + x64*(x65 + x68)) + radial_eval_alpha*x59*x7);
      basis_x_eval[ipt + 12*npts] = x60*(x61*x71 + x89*(x78 + x87 - x88));

      // Evaluate first derivative of bfn wrt y
      basis_y_eval[ipt + 0*npts] = 0.0625*x0*(radial_eval*(x12 + x90*(x92 + x94)) + radial_eval_alpha*x12*x8);
      basis_y_eval[ipt + 1*npts] = x14*(radial_eval*(x17 + x90*(x91 + x94)) + radial_eval_alpha*x17*x8);
      basis_y_eval[ipt + 2*npts] = x*x18*(radial_eval*(x25 - x90*(x22*x93 - x91)) + radial_eval_alpha*x25*x8);
      basis_y_eval[ipt + 3*npts] = x86*(-radial_eval*(x90*(3.0*x7*y - 3.0*x91 + x95) - x97) + radial_eval_alpha*x30*x8);
      basis_y_eval[ipt + 4*npts] = x98*(radial_eval*(x33 + x90*(x101 - x22*x99)) + radial_eval_alpha*x33*x8);
      basis_y_eval[ipt + 5*npts] = x35*(radial_eval*(20.0*x102*y + x38) + radial_eval_alpha*x38*x8);
      basis_y_eval[ipt + 6*npts] = 0.0625*x107*x47 - 0.0625*x77*(x103 + x104 + x106 - 12.0*x22*x7*y - 12.0*x22*x91 + x31*x99);
      basis_y_eval[ipt + 7*npts] = x48*(x102*x69 + x107*x38);
      basis_y_eval[ipt + 8*npts] = x49*(x107*x54 - x82*(-x104 + x106 + x108 - 32.0*x22*x91 + 16.0*x31*y));
      basis_y_eval[ipt + 9*npts] = x*x86*(x107*x55 + x73*(x100 + x92 - x95));
      basis_y_eval[ipt + 10*npts] = x56*(x107*x57 + x82*(10.0*x105 - x108 + x109 + 20.0*x22*x91 - 60.0*x23*y));
      basis_y_eval[ipt + 11*npts] = x58*(x107*x59 - x69*(x100 - x91));
      basis_y_eval[ipt + 12*npts] = x60*(x107*x61 - x89*(x103 + x109 - 10.0*x7*x91));

      // Evaluate first derivative of bfn wrt z
      basis_z_eval[ipt + 0*npts] = x12*x63*x71*z;
      basis_z_eval[ipt + 1*npts] = x13*x17*y*(radial_eval + radial_eval_alpha*x22);
      basis_z_eval[ipt + 2*npts] = x20*z*(radial_eval_alpha*x25 + x69*(x110 + x7));
      basis_z_eval[ipt + 3*npts] = x72*(radial_eval*(x111*(x110 + 3.0*x7) + x97) + radial_eval_alpha*x22*x30);
      basis_z_eval[ipt + 4*npts] = -x*x72*(x112*(-2.0*x113 + x7*z + x8*z) - x114*x33);
      basis_z_eval[ipt + 5*npts] = x116*x34*y;
      basis_z_eval[ipt + 6*npts] = 0.0625*x114*x47 + 0.0625*x73*(x117*x3 + x117*x5 - 40.0*x118 - 40.0*x119 + 30.0*x9*z + 8.0*(z*z*z*z*z));
      basis_z_eval[ipt + 7*npts] = x*x116*x34;
      basis_z_eval[ipt + 8*npts] = x49*(x112*(2.0*x118 - 2.0*x119 - x3*z + x5*z) + x114*x54);
      basis_z_eval[ipt + 9*npts] = x98*(radial_eval*(x111*(x7 - 3.0*x8) + x55) + radial_eval_alpha*x22*x55);
      basis_z_eval[ipt + 10*npts] = x56*z*(radial_eval_alpha*x57 + x69*(x32 + x96));
      basis_z_eval[ipt + 11*npts] = x*x13*x59*(radial_eval + radial_eval_alpha*x22);
      basis_z_eval[ipt + 12*npts] = x114*x60*x61;







#if 0
      // Evaluate the angular part of bfn



      double ang_eval_0;
      double ang_eval_1;
      double ang_eval_2;
      double ang_eval_3;


      ang_eval_0 = x0*x12*x2;
      ang_eval_1 = radial_eval*x15*x17;
      ang_eval_2 = radial_eval*x20*x25;
      ang_eval_3 = x26*x30*z;
      basis_eval[ipt + 0*npts] = ang_eval_0;
      basis_eval[ipt + 1*npts] = ang_eval_1;
      basis_eval[ipt + 2*npts] = ang_eval_2;
      basis_eval[ipt + 3*npts] = ang_eval_3;

      ang_eval_0 = x*x26*x33;
      ang_eval_1 = radial_eval*x36*x38;
      ang_eval_2 = x1*x47;
      ang_eval_3 = radial_eval*x38*x48;
      basis_eval[ipt + 4*npts] = ang_eval_0;
      basis_eval[ipt + 5*npts] = ang_eval_1;
      basis_eval[ipt + 6*npts] = ang_eval_2;
      basis_eval[ipt + 7*npts] = ang_eval_3;

      ang_eval_0 = radial_eval*x49*x54;
      ang_eval_1 = sqrt_210*x*x1*x55*z;
      ang_eval_2 = radial_eval*x56*x57;
      ang_eval_3 = radial_eval*x58*x59;
      basis_eval[ipt + 8*npts] = ang_eval_0;
      basis_eval[ipt + 9*npts] = ang_eval_1;
      basis_eval[ipt + 10*npts] = ang_eval_2;
      basis_eval[ipt + 11*npts] = ang_eval_3;

      ang_eval_0 = radial_eval*x60*x61;
      basis_eval[ipt + 12*npts] = ang_eval_0;


      double dang_eval_x_0, dang_eval_y_0, dang_eval_z_0;
      double dang_eval_x_1, dang_eval_y_1, dang_eval_z_1;
      double dang_eval_x_2, dang_eval_y_2, dang_eval_z_2;
      double dang_eval_x_3, dang_eval_y_3, dang_eval_z_3;

      dang_eval_x_0 = x63*(radial_eval*(x12 + x64*(x66 + x68)) + radial_eval_alpha*x12*x7);
      dang_eval_y_0 = 0.0625*x0*(radial_eval*(x12 + x90*(x92 + x94)) + radial_eval_alpha*x12*x8);
      dang_eval_z_0 = x12*x63*x71*z;
      dang_eval_x_1 = x15*(x17*x71 + x69*(x65 - x70));
      dang_eval_y_1 = x14*(radial_eval*(x17 + x90*(x91 + x94)) + radial_eval_alpha*x17*x8);
      dang_eval_z_1 = x13*x17*y*(radial_eval + radial_eval_alpha*x22);
      dang_eval_x_2 = x19*(radial_eval*(x25 + x64*(x22*x67 - x65)) + radial_eval_alpha*x25*x7);
      dang_eval_y_2 = x*x18*(radial_eval*(x25 - x90*(x22*x93 - x91)) + radial_eval_alpha*x25*x8);
      dang_eval_z_2 = x20*z*(radial_eval_alpha*x25 + x69*(x110 + x7));
      dang_eval_x_3 = x72*z*(x30*x71 - x73*(-4.0*x*x22 + x66 + x70));
      dang_eval_y_3 = x86*(-radial_eval*(x90*(3.0*x7*y - 3.0*x91 + x95) - x97) + radial_eval_alpha*x30*x8);
      dang_eval_z_3 = x72*(radial_eval*(x111*(x110 + 3.0*x7) + x97) + radial_eval_alpha*x22*x30);
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

      dang_eval_x_0 = x72*(radial_eval*(x33 - x64*(x22*x74 - x75)) + radial_eval_alpha*x33*x7);
      dang_eval_y_0 = x98*(radial_eval*(x33 + x90*(x101 - x22*x99)) + radial_eval_alpha*x33*x8);
      dang_eval_z_0 = -x*x72*(x112*(-2.0*x113 + x7*z + x8*z) - x114*x33);
      dang_eval_x_1 = x36*(x38*x71 + x69*x76);
      dang_eval_y_1 = x35*(radial_eval*(20.0*x102*y + x38) + radial_eval_alpha*x38*x8);
      dang_eval_z_1 = x116*x34*y;
      dang_eval_x_2 = 0.0625*x47*x71 - 0.0625*x77*(-12.0*x*x22*x8 - 12.0*x22*x65 + x31*x74 + x78 + x79 + x81);
      dang_eval_y_2 = 0.0625*x107*x47 - 0.0625*x77*(x103 + x104 + x106 - 12.0*x22*x7*y - 12.0*x22*x91 + x31*x99);
      dang_eval_z_2 = 0.0625*x114*x47 + 0.0625*x73*(x117*x3 + x117*x5 - 40.0*x118 - 40.0*x119 + 30.0*x9*z + 8.0*(z*z*z*z*z));
      dang_eval_x_3 = x35*(radial_eval*(20.0*x*x76 + x38) + radial_eval_alpha*x38*x7);
      dang_eval_y_3 = x48*(x102*x69 + x107*x38);
      dang_eval_z_3 = x*x116*x34;
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

      dang_eval_x_0 = x49*(x54*x71 + x82*(16.0*x*x31 - x79 + x81 + x83 - 32.0*x84));
      dang_eval_y_0 = x49*(x107*x54 - x82*(-x104 + x106 + x108 - 32.0*x22*x91 + 16.0*x31*y));
      dang_eval_z_0 = x49*(x112*(2.0*x118 - 2.0*x119 - x3*z + x5*z) + x114*x54);
      dang_eval_x_1 = x86*(radial_eval*(x55 + x64*(3.0*x*x8 + x22*x64 - x66)) + radial_eval_alpha*x55*x7);
      dang_eval_y_1 = x*x86*(x107*x55 + x73*(x100 + x92 - x95));
      dang_eval_z_1 = x98*(radial_eval*(x111*(x7 - 3.0*x8) + x55) + radial_eval_alpha*x22*x55);
      dang_eval_x_2 = x56*(x57*x71 + x82*(-60.0*x*x24 - x83 + 20.0*x84 + x87 + x88));
      dang_eval_y_2 = x56*(x107*x57 + x82*(10.0*x105 - x108 + x109 + 20.0*x22*x91 - 60.0*x23*y));
      dang_eval_z_2 = x56*z*(radial_eval_alpha*x57 + x69*(x32 + x96));
      dang_eval_x_3 = x14*(radial_eval*(x59 + x64*(x65 + x68)) + radial_eval_alpha*x59*x7);
      dang_eval_y_3 = x58*(x107*x59 - x69*(x100 - x91));
      dang_eval_z_3 = x*x13*x59*(radial_eval + radial_eval_alpha*x22);
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

      dang_eval_x_0 = x60*(x61*x71 + x89*(x78 + x87 - x88));
      dang_eval_y_0 = x60*(x107*x61 - x89*(x103 + x109 - 10.0*x7*x91));
      dang_eval_z_0 = x114*x60*x61;
      basis_x_eval[ipt + 12*npts] = dang_eval_x_0;
      basis_y_eval[ipt + 12*npts] = dang_eval_y_0;
      basis_z_eval[ipt + 12*npts] = dang_eval_z_0;

#endif
    } // Loop over points within task
  } // Loop over tasks
        
  } // Loop over shells
} // end kernel

} // namespace GauXC
