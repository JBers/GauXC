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


__global__ __launch_bounds__(512,2) void collocation_device_shell_to_task_kernel_spherical_6(
  uint32_t                        nshell,
  ShellToTaskDevice* __restrict__ shell_to_task,
  XCDeviceTask*      __restrict__ device_tasks
) {


  __shared__ double alpha[512/cuda::warp_size][detail::shell_nprim_max + 1]; 
  __shared__ double coeff[512/cuda::warp_size][detail::shell_nprim_max + 1];
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

      #pragma unroll 1
      for( uint32_t i = 0; i < nprim; ++i ) {
        const auto a = my_alpha[i];
        const auto e = my_coeff[i] * std::exp( - a * rsq );

        radial_eval += e;
      }


      // Common Subexpressions
      const auto x0 = 0.0625*radial_eval; 
      const auto x1 = x*y; 
      const auto x2 = x0*x1; 
      const auto x3 = x*x*x*x; 
      const auto x4 = 3.0*x3; 
      const auto x5 = y*y*y*y; 
      const auto x6 = x*x; 
      const auto x7 = y*y; 
      const auto x8 = x6*x7; 
      const auto x9 = 10.0*x8; 
      const auto x10 = -x9; 
      const auto x11 = y*z; 
      const auto x12 = 0.1875*radial_eval; 
      const auto x13 = sqrt_154*x12; 
      const auto x14 = 5.0*x3; 
      const auto x15 = z*z; 
      const auto x16 = x15*x6; 
      const auto x17 = x15*x7; 
      const auto x18 = sqrt_210*x0; 
      const auto x19 = 6.0*x8; 
      const auto x20 = z*z*z*z; 
      const auto x21 = 0.125*radial_eval*sqrt_21; 
      const auto x22 = 5.0*x5; 
      const auto x23 = x14 - 20.0*x16 - 20.0*x17 + 8.0*x20 + x22 + x9; 
      const auto x24 = x*x*x*x*x*x; 
      const auto x25 = y*y*y*y*y*y; 
      const auto x26 = x5*x6; 
      const auto x27 = 15.0*x26; 
      const auto x28 = x20*x6; 
      const auto x29 = x3*x7; 
      const auto x30 = 15.0*x29; 
      const auto x31 = x20*x7; 
      const auto x32 = x*z; 
      const auto x33 = 0.03125*radial_eval; 
      const auto x34 = x15*x3; 
      const auto x35 = x15*x5; 
      const auto x36 = -x25; 
      const auto x37 = x24 + x36; 


      // Evaluate basis function
      basis_eval[ipt + 0*npts] = sqrt_462*x2*(x10 + x4 + 3.0*x5);
      basis_eval[ipt + 1*npts] = x11*x13*(x10 + x14 + x5);
      basis_eval[ipt + 2*npts] = 0.75*radial_eval*sqrt_7*x1*(10.0*x16 - 10.0*x17 - x3 + x5);
      basis_eval[ipt + 3*npts] = -x11*x18*(-24.0*x15*x6 + 8.0*x17 + x19 + 9.0*x3 - 3.0*x5);
      basis_eval[ipt + 4*npts] = sqrt_210*x2*(-16.0*x16 - 16.0*x17 + 16.0*x20 + x3 + x5 + 2.0*x8);
      basis_eval[ipt + 5*npts] = x11*x21*x23;
      basis_eval[ipt + 6*npts] = -x0*(-90.0*x15*x3 - 90.0*x15*x5 - 180.0*x15*x6*x7 + 5.0*x24 + 5.0*x25 + x27 + 120.0*x28 + x30 + 120.0*x31 - 16.0*z*z*z*z*z*z);
      basis_eval[ipt + 7*npts] = x21*x23*x32;
      basis_eval[ipt + 8*npts] = sqrt_210*x33*(-x26 + 16.0*x28 + x29 - 16.0*x31 - 16.0*x34 + 16.0*x35 + x37);
      basis_eval[ipt + 9*npts] = x18*x32*(8.0*x16 - 24.0*x17 + x19 - x4 + 9.0*x5);
      basis_eval[ipt + 10*npts] = sqrt_7*x12*(-60.0*x15*x6*x7 - x24 + 5.0*x26 + 5.0*x29 + 10.0*x34 + 10.0*x35 + x36);
      basis_eval[ipt + 11*npts] = x13*x32*(x10 + x22 + x3);
      basis_eval[ipt + 12*npts] = sqrt_462*x33*(x27 - x30 + x37);


    







#if 0
      // Evaluate the angular part of bfn



      double ang_eval_0;
      double ang_eval_1;
      double ang_eval_2;
      double ang_eval_3;


      ang_eval_0 = sqrt_462*x2*(x10 + x4 + 3.0*x5);
      ang_eval_1 = x11*x13*(x10 + x14 + x5);
      ang_eval_2 = 0.75*radial_eval*sqrt_7*x1*(10.0*x16 - 10.0*x17 - x3 + x5);
      ang_eval_3 = -x11*x18*(-24.0*x15*x6 + 8.0*x17 + x19 + 9.0*x3 - 3.0*x5);
      basis_eval[ipt + 0*npts] = ang_eval_0;
      basis_eval[ipt + 1*npts] = ang_eval_1;
      basis_eval[ipt + 2*npts] = ang_eval_2;
      basis_eval[ipt + 3*npts] = ang_eval_3;

      ang_eval_0 = sqrt_210*x2*(-16.0*x16 - 16.0*x17 + 16.0*x20 + x3 + x5 + 2.0*x8);
      ang_eval_1 = x11*x21*x23;
      ang_eval_2 = -x0*(-90.0*x15*x3 - 90.0*x15*x5 - 180.0*x15*x6*x7 + 5.0*x24 + 5.0*x25 + x27 + 120.0*x28 + x30 + 120.0*x31 - 16.0*z*z*z*z*z*z);
      ang_eval_3 = x21*x23*x32;
      basis_eval[ipt + 4*npts] = ang_eval_0;
      basis_eval[ipt + 5*npts] = ang_eval_1;
      basis_eval[ipt + 6*npts] = ang_eval_2;
      basis_eval[ipt + 7*npts] = ang_eval_3;

      ang_eval_0 = sqrt_210*x33*(-x26 + 16.0*x28 + x29 - 16.0*x31 - 16.0*x34 + 16.0*x35 + x37);
      ang_eval_1 = x18*x32*(8.0*x16 - 24.0*x17 + x19 - x4 + 9.0*x5);
      ang_eval_2 = sqrt_7*x12*(-60.0*x15*x6*x7 - x24 + 5.0*x26 + 5.0*x29 + 10.0*x34 + 10.0*x35 + x36);
      ang_eval_3 = x13*x32*(x10 + x22 + x3);
      basis_eval[ipt + 8*npts] = ang_eval_0;
      basis_eval[ipt + 9*npts] = ang_eval_1;
      basis_eval[ipt + 10*npts] = ang_eval_2;
      basis_eval[ipt + 11*npts] = ang_eval_3;

      ang_eval_0 = sqrt_462*x33*(x27 - x30 + x37);
      basis_eval[ipt + 12*npts] = ang_eval_0;


#endif
    } // Loop over points within task
  } // Loop over tasks
        
  } // Loop over shells
} // end kernel

} // namespace GauXC
