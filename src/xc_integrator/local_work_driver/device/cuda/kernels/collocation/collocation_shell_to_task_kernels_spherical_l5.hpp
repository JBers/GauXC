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


__global__ __launch_bounds__(512,2) void collocation_device_shell_to_task_kernel_spherical_5(
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
      const auto x0 = radial_eval*y; 
      const auto x1 = 0.1875*sqrt_14; 
      const auto x2 = y*y*y*y; 
      const auto x3 = x*x*x*x; 
      const auto x4 = x*x; 
      const auto x5 = y*y; 
      const auto x6 = x4*x5; 
      const auto x7 = -10.0*x6; 
      const auto x8 = x*x0*z; 
      const auto x9 = 0.0625*sqrt_70; 
      const auto x10 = 2.0*x6; 
      const auto x11 = z*z; 
      const auto x12 = x11*x5; 
      const auto x13 = 0.125*radial_eval; 
      const auto x14 = sqrt_15*x13; 
      const auto x15 = 8.0*(z*z*z*z); 
      const auto x16 = x11*x4; 
      const auto x17 = x2 + x3; 
      const auto x18 = x10 - 12.0*x12 + x15 - 12.0*x16 + x17; 
      const auto x19 = radial_eval*z; 
      const auto x20 = -x3; 
      const auto x21 = radial_eval*x; 


      // Evaluate basis function
      basis_eval[ipt + 0*npts] = x0*x1*(x2 + 5.0*x3 + x7);
      basis_eval[ipt + 1*npts] = 1.5*sqrt_35*x8*(x4 - x5);
      basis_eval[ipt + 2*npts] = -x0*x9*(x10 - 24.0*x11*x4 + 8.0*x12 - x2 + 3.0*x3);
      basis_eval[ipt + 3*npts] = -0.5*sqrt_105*x8*(-2.0*x11 + x4 + x5);
      basis_eval[ipt + 4*npts] = x14*x18*y;
      basis_eval[ipt + 5*npts] = x13*z*(-40.0*x12 + x15 - 40.0*x16 + 15.0*x2 + 15.0*x3 + 30.0*x6);
      basis_eval[ipt + 6*npts] = x*x14*x18;
      basis_eval[ipt + 7*npts] = 0.25*sqrt_105*x19*(-2.0*x12 + 2.0*x16 + x2 + x20);
      basis_eval[ipt + 8*npts] = x21*x9*(x10 - 24.0*x12 + 8.0*x16 + 3.0*x2 + x20);
      basis_eval[ipt + 9*npts] = 0.375*sqrt_35*x19*(x17 - 6.0*x6);
      basis_eval[ipt + 10*npts] = x1*x21*(5.0*x2 + x3 + x7);


    







#if 0
      // Evaluate the angular part of bfn



      double ang_eval_0;
      double ang_eval_1;
      double ang_eval_2;
      double ang_eval_3;


      ang_eval_0 = x0*x1*(x2 + 5.0*x3 + x7);
      ang_eval_1 = 1.5*sqrt_35*x8*(x4 - x5);
      ang_eval_2 = -x0*x9*(x10 - 24.0*x11*x4 + 8.0*x12 - x2 + 3.0*x3);
      ang_eval_3 = -0.5*sqrt_105*x8*(-2.0*x11 + x4 + x5);
      basis_eval[ipt + 0*npts] = ang_eval_0;
      basis_eval[ipt + 1*npts] = ang_eval_1;
      basis_eval[ipt + 2*npts] = ang_eval_2;
      basis_eval[ipt + 3*npts] = ang_eval_3;

      ang_eval_0 = x14*x18*y;
      ang_eval_1 = x13*z*(-40.0*x12 + x15 - 40.0*x16 + 15.0*x2 + 15.0*x3 + 30.0*x6);
      ang_eval_2 = x*x14*x18;
      ang_eval_3 = 0.25*sqrt_105*x19*(-2.0*x12 + 2.0*x16 + x2 + x20);
      basis_eval[ipt + 4*npts] = ang_eval_0;
      basis_eval[ipt + 5*npts] = ang_eval_1;
      basis_eval[ipt + 6*npts] = ang_eval_2;
      basis_eval[ipt + 7*npts] = ang_eval_3;

      ang_eval_0 = x21*x9*(x10 - 24.0*x12 + 8.0*x16 + 3.0*x2 + x20);
      ang_eval_1 = 0.375*sqrt_35*x19*(x17 - 6.0*x6);
      ang_eval_2 = x1*x21*(5.0*x2 + x3 + x7);
      basis_eval[ipt + 8*npts] = ang_eval_0;
      basis_eval[ipt + 9*npts] = ang_eval_1;
      basis_eval[ipt + 10*npts] = ang_eval_2;


#endif
    } // Loop over points within task
  } // Loop over tasks
        
  } // Loop over shells
} // end kernel

} // namespace GauXC
