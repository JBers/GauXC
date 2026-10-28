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


__global__ __launch_bounds__(128,2) void collocation_device_shell_to_task_kernel_spherical_gradient_5(
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
      const auto x0 = 0.1875*sqrt_14; 
      const auto x1 = x0*y; 
      const auto x2 = y*y*y*y; 
      const auto x3 = x*x*x*x; 
      const auto x4 = x*x; 
      const auto x5 = y*y; 
      const auto x6 = x4*x5; 
      const auto x7 = -10.0*x6; 
      const auto x8 = x2 + 5.0*x3 + x7; 
      const auto x9 = y*z; 
      const auto x10 = 1.5*sqrt_35; 
      const auto x11 = x10*x9; 
      const auto x12 = radial_eval*x; 
      const auto x13 = -x5; 
      const auto x14 = x13 + x4; 
      const auto x15 = 0.0625*sqrt_70; 
      const auto x16 = x15*y; 
      const auto x17 = 3.0*x3; 
      const auto x18 = 2.0*x6; 
      const auto x19 = z*z; 
      const auto x20 = x19; 
      const auto x21 = x20*x5; 
      const auto x22 = 8.0*x21; 
      const auto x23 = -x17 - x18 + x2 + 24.0*x20*x4 - x22; 
      const auto x24 = 0.5*sqrt_105; 
      const auto x25 = x24*x9; 
      const auto x26 = -2.0*x20; 
      const auto x27 = x26 + x5; 
      const auto x28 = -x27 - x4; 
      const auto x29 = 0.125*sqrt_15; 
      const auto x30 = x29*y; 
      const auto x31 = 8.0*(z*z*z*z); 
      const auto x32 = x20*x4; 
      const auto x33 = x2 + x3; 
      const auto x34 = x18 - 12.0*x21 + x31 - 12.0*x32 + x33; 
      const auto x35 = 0.125*z; 
      const auto x36 = 15.0*x2 - 40.0*x21 + 15.0*x3 + x31 - 40.0*x32 + 30.0*x6; 
      const auto x37 = x*x29; 
      const auto x38 = 0.25*sqrt_105; 
      const auto x39 = x38*z; 
      const auto x40 = -x3; 
      const auto x41 = x2 - 2.0*x21 + 2.0*x32 + x40; 
      const auto x42 = x*x15; 
      const auto x43 = x18 + 3.0*x2 - 24.0*x21 + 8.0*x32 + x40; 
      const auto x44 = 0.375*sqrt_35; 
      const auto x45 = x44*z; 
      const auto x46 = x33 - 6.0*x6; 
      const auto x47 = x*x0; 
      const auto x48 = 5.0*x2 + x3 + x7; 
      const auto x49 = 20.0*radial_eval; 
      const auto x50 = x*x*x; 
      const auto x51 = x*x5; 
      const auto x52 = radial_eval_alpha*x; 
      const auto x53 = 3.0*x4; 
      const auto x54 = x13 + x53; 
      const auto x55 = 4.0*radial_eval; 
      const auto x56 = 3.0*x50; 
      const auto x57 = -6.0*x*x20 + x50 + x51; 
      const auto x58 = 3.0*x*x5; 
      const auto x59 = 4.0*x; 
      const auto x60 = x20*x59; 
      const auto x61 = -x50; 
      const auto x62 = 4.0*y; 
      const auto x63 = y*y*y; 
      const auto x64 = x*z; 
      const auto x65 = 3.0*x5; 
      const auto x66 = x4 - x65; 
      const auto x67 = x4*y; 
      const auto x68 = x20*x62; 
      const auto x69 = -6.0*x20*y + x63 + x67; 
      const auto x70 = 3.0*x63; 
      const auto x71 = 3.0*x4*y; 
      const auto x72 = radial_eval_alpha*y; 
      const auto x73 = x0*z; 
      const auto x74 = x*y; 
      const auto x75 = 16.0*radial_eval; 
      const auto x76 = z*z*z; 
      const auto x77 = 3.0*z; 
      const auto x78 = -8.0*radial_eval*(x4*x77 + x5*x77 - 4.0*x76) + radial_eval_alpha*x34*z; 
      const auto x79 = 5.0*z; 


      // Evaluate basis function
      basis_eval[ipt + 0*npts] = radial_eval*x1*x8;
      basis_eval[ipt + 1*npts] = x11*x12*x14;
      basis_eval[ipt + 2*npts] = radial_eval*x16*x23;
      basis_eval[ipt + 3*npts] = x12*x25*x28;
      basis_eval[ipt + 4*npts] = radial_eval*x30*x34;
      basis_eval[ipt + 5*npts] = radial_eval*x35*x36;
      basis_eval[ipt + 6*npts] = radial_eval*x34*x37;
      basis_eval[ipt + 7*npts] = radial_eval*x39*x41;
      basis_eval[ipt + 8*npts] = radial_eval*x42*x43;
      basis_eval[ipt + 9*npts] = radial_eval*x45*x46;
      basis_eval[ipt + 10*npts] = radial_eval*x47*x48;


    
      // Evaluate first derivative of bfn wrt x
      basis_x_eval[ipt + 0*npts] = x1*(x49*(x50 - x51) + x52*x8);
      basis_x_eval[ipt + 1*npts] = x11*(radial_eval*x54 + radial_eval_alpha*x14*x4);
      basis_x_eval[ipt + 2*npts] = x16*(x23*x52 - x55*(-12.0*x*x20 + x51 + x56));
      basis_x_eval[ipt + 3*npts] = -x25*(radial_eval*(x27 + x53) - radial_eval_alpha*x28*x4);
      basis_x_eval[ipt + 4*npts] = x30*(x34*x52 + x55*x57);
      basis_x_eval[ipt + 5*npts] = x35*(x36*x52 + x49*(x56 + x58 - x60));
      basis_x_eval[ipt + 6*npts] = x29*(radial_eval*(x34 + x57*x59) + radial_eval_alpha*x34*x4);
      basis_x_eval[ipt + 7*npts] = x39*(x41*x52 + x55*(x*x20 + x61));
      basis_x_eval[ipt + 8*npts] = x15*(radial_eval*(x43 + x59*(x51 + x60 + x61)) + radial_eval_alpha*x4*x43);
      basis_x_eval[ipt + 9*npts] = x45*(x46*x52 + x55*(x50 - x58));
      basis_x_eval[ipt + 10*npts] = x0*(radial_eval*(x48 - x59*(5.0*x*x5 - x50)) + radial_eval_alpha*x4*x48);

      // Evaluate first derivative of bfn wrt y
      basis_y_eval[ipt + 0*npts] = x0*(-radial_eval*(x62*(5.0*x4*y - x63) - x8) + radial_eval_alpha*x5*x8);
      basis_y_eval[ipt + 1*npts] = x10*x64*(radial_eval*x66 + radial_eval_alpha*x14*x5);
      basis_y_eval[ipt + 2*npts] = -x15*(radial_eval*(x17 + x18 - x2 + x22 - 24.0*x32 + x62*(-x63 + x67 + x68)) - radial_eval_alpha*x23*x5);
      basis_y_eval[ipt + 3*npts] = -x24*x64*(radial_eval*(x26 + x4 + x65) - radial_eval_alpha*x28*x5);
      basis_y_eval[ipt + 4*npts] = x29*(radial_eval*(x34 + x62*x69) + radial_eval_alpha*x34*x5);
      basis_y_eval[ipt + 5*npts] = x35*(x36*x72 + x49*(-x68 + x70 + x71));
      basis_y_eval[ipt + 6*npts] = x37*(x34*x72 + x55*x69);
      basis_y_eval[ipt + 7*npts] = x39*(x41*x72 - x55*(x20*y - x63));
      basis_y_eval[ipt + 8*npts] = x42*(x43*x72 + x55*(-12.0*x20*y + x67 + x70));
      basis_y_eval[ipt + 9*npts] = x45*(x46*x72 + x55*(x63 - x71));
      basis_y_eval[ipt + 10*npts] = x47*(x48*x72 + x49*(x63 - x67));

      // Evaluate first derivative of bfn wrt z
      basis_z_eval[ipt + 0*npts] = x72*x73*x8;
      basis_z_eval[ipt + 1*npts] = x10*x14*x74*(radial_eval + radial_eval_alpha*x20);
      basis_z_eval[ipt + 2*npts] = x16*z*(radial_eval_alpha*x23 + x54*x75);
      basis_z_eval[ipt + 3*npts] = -x24*x74*(radial_eval*(-6.0*x20 + x4 + x5) - radial_eval_alpha*x20*x28);
      basis_z_eval[ipt + 4*npts] = x30*x78;
      basis_z_eval[ipt + 5*npts] = 0.125*radial_eval*(x36 - 16.0*z*(x4*x79 + x5*x79 - 2.0*x76)) + 0.125*radial_eval_alpha*x20*x36;
      basis_z_eval[ipt + 6*npts] = x37*x78;
      basis_z_eval[ipt + 7*npts] = x38*(radial_eval*(4.0*x14*x19 + x41) + radial_eval_alpha*x20*x41);
      basis_z_eval[ipt + 8*npts] = x42*z*(radial_eval_alpha*x43 + x66*x75);
      basis_z_eval[ipt + 9*npts] = x44*x46*(radial_eval + radial_eval_alpha*x20);
      basis_z_eval[ipt + 10*npts] = x48*x52*x73;







#if 0
      // Evaluate the angular part of bfn



      double ang_eval_0;
      double ang_eval_1;
      double ang_eval_2;
      double ang_eval_3;


      ang_eval_0 = radial_eval*x1*x8;
      ang_eval_1 = x11*x12*x14;
      ang_eval_2 = radial_eval*x16*x23;
      ang_eval_3 = x12*x25*x28;
      basis_eval[ipt + 0*npts] = ang_eval_0;
      basis_eval[ipt + 1*npts] = ang_eval_1;
      basis_eval[ipt + 2*npts] = ang_eval_2;
      basis_eval[ipt + 3*npts] = ang_eval_3;

      ang_eval_0 = radial_eval*x30*x34;
      ang_eval_1 = radial_eval*x35*x36;
      ang_eval_2 = radial_eval*x34*x37;
      ang_eval_3 = radial_eval*x39*x41;
      basis_eval[ipt + 4*npts] = ang_eval_0;
      basis_eval[ipt + 5*npts] = ang_eval_1;
      basis_eval[ipt + 6*npts] = ang_eval_2;
      basis_eval[ipt + 7*npts] = ang_eval_3;

      ang_eval_0 = radial_eval*x42*x43;
      ang_eval_1 = radial_eval*x45*x46;
      ang_eval_2 = radial_eval*x47*x48;
      basis_eval[ipt + 8*npts] = ang_eval_0;
      basis_eval[ipt + 9*npts] = ang_eval_1;
      basis_eval[ipt + 10*npts] = ang_eval_2;


      double dang_eval_x_0, dang_eval_y_0, dang_eval_z_0;
      double dang_eval_x_1, dang_eval_y_1, dang_eval_z_1;
      double dang_eval_x_2, dang_eval_y_2, dang_eval_z_2;
      double dang_eval_x_3, dang_eval_y_3, dang_eval_z_3;

      dang_eval_x_0 = x1*(x49*(x50 - x51) + x52*x8);
      dang_eval_y_0 = x0*(-radial_eval*(x62*(5.0*x4*y - x63) - x8) + radial_eval_alpha*x5*x8);
      dang_eval_z_0 = x72*x73*x8;
      dang_eval_x_1 = x11*(radial_eval*x54 + radial_eval_alpha*x14*x4);
      dang_eval_y_1 = x10*x64*(radial_eval*x66 + radial_eval_alpha*x14*x5);
      dang_eval_z_1 = x10*x14*x74*(radial_eval + radial_eval_alpha*x20);
      dang_eval_x_2 = x16*(x23*x52 - x55*(-12.0*x*x20 + x51 + x56));
      dang_eval_y_2 = -x15*(radial_eval*(x17 + x18 - x2 + x22 - 24.0*x32 + x62*(-x63 + x67 + x68)) - radial_eval_alpha*x23*x5);
      dang_eval_z_2 = x16*z*(radial_eval_alpha*x23 + x54*x75);
      dang_eval_x_3 = -x25*(radial_eval*(x27 + x53) - radial_eval_alpha*x28*x4);
      dang_eval_y_3 = -x24*x64*(radial_eval*(x26 + x4 + x65) - radial_eval_alpha*x28*x5);
      dang_eval_z_3 = -x24*x74*(radial_eval*(-6.0*x20 + x4 + x5) - radial_eval_alpha*x20*x28);
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

      dang_eval_x_0 = x30*(x34*x52 + x55*x57);
      dang_eval_y_0 = x29*(radial_eval*(x34 + x62*x69) + radial_eval_alpha*x34*x5);
      dang_eval_z_0 = x30*x78;
      dang_eval_x_1 = x35*(x36*x52 + x49*(x56 + x58 - x60));
      dang_eval_y_1 = x35*(x36*x72 + x49*(-x68 + x70 + x71));
      dang_eval_z_1 = 0.125*radial_eval*(x36 - 16.0*z*(x4*x79 + x5*x79 - 2.0*x76)) + 0.125*radial_eval_alpha*x20*x36;
      dang_eval_x_2 = x29*(radial_eval*(x34 + x57*x59) + radial_eval_alpha*x34*x4);
      dang_eval_y_2 = x37*(x34*x72 + x55*x69);
      dang_eval_z_2 = x37*x78;
      dang_eval_x_3 = x39*(x41*x52 + x55*(x*x20 + x61));
      dang_eval_y_3 = x39*(x41*x72 - x55*(x20*y - x63));
      dang_eval_z_3 = x38*(radial_eval*(4.0*x14*x19 + x41) + radial_eval_alpha*x20*x41);
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

      dang_eval_x_0 = x15*(radial_eval*(x43 + x59*(x51 + x60 + x61)) + radial_eval_alpha*x4*x43);
      dang_eval_y_0 = x42*(x43*x72 + x55*(-12.0*x20*y + x67 + x70));
      dang_eval_z_0 = x42*z*(radial_eval_alpha*x43 + x66*x75);
      dang_eval_x_1 = x45*(x46*x52 + x55*(x50 - x58));
      dang_eval_y_1 = x45*(x46*x72 + x55*(x63 - x71));
      dang_eval_z_1 = x44*x46*(radial_eval + radial_eval_alpha*x20);
      dang_eval_x_2 = x0*(radial_eval*(x48 - x59*(5.0*x*x5 - x50)) + radial_eval_alpha*x4*x48);
      dang_eval_y_2 = x47*(x48*x72 + x49*(x63 - x67));
      dang_eval_z_2 = x48*x52*x73;
      basis_x_eval[ipt + 8*npts] = dang_eval_x_0;
      basis_y_eval[ipt + 8*npts] = dang_eval_y_0;
      basis_z_eval[ipt + 8*npts] = dang_eval_z_0;
      basis_x_eval[ipt + 9*npts] = dang_eval_x_1;
      basis_y_eval[ipt + 9*npts] = dang_eval_y_1;
      basis_z_eval[ipt + 9*npts] = dang_eval_z_1;
      basis_x_eval[ipt + 10*npts] = dang_eval_x_2;
      basis_y_eval[ipt + 10*npts] = dang_eval_y_2;
      basis_z_eval[ipt + 10*npts] = dang_eval_z_2;

#endif
    } // Loop over points within task
  } // Loop over tasks
        
  } // Loop over shells
} // end kernel

} // namespace GauXC
