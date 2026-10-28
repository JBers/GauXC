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


__global__ __launch_bounds__(128,2) void collocation_device_shell_to_task_kernel_cartesian_laplacian_5(
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
    auto* __restrict__ basis_lapl_eval = task->d2bflapl + shoff;

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
      double radial_eval_alpha_squared = 0.;

      #pragma unroll 1
      for( uint32_t i = 0; i < nprim; ++i ) {
        const auto a = my_alpha[i];
        const auto e = my_coeff[i] * std::exp( - a * rsq );

        radial_eval += e;
        radial_eval_alpha += a * e;
        radial_eval_alpha_squared += a * a * e;
      }

      radial_eval_alpha *= -2;
      radial_eval_alpha_squared *= 4;

      // Common Subexpressions
      const auto x0 = x*x*x*x*x; 
      const auto x1 = radial_eval*y; 
      const auto x2 = x*x*x*x; 
      const auto x3 = radial_eval*z; 
      const auto x4 = x*x*x; 
      const auto x5 = y*y; 
      const auto x6 = x4*x5; 
      const auto x7 = x1*z; 
      const auto x8 = z*z; 
      const auto x9 = x4*x8; 
      const auto x10 = x*x; 
      const auto x11 = y*y*y; 
      const auto x12 = x10*x11; 
      const auto x13 = x10*x5; 
      const auto x14 = x10*x8; 
      const auto x15 = z*z*z; 
      const auto x16 = x10*x15; 
      const auto x17 = radial_eval*x; 
      const auto x18 = y*y*y*y; 
      const auto x19 = x*x3; 
      const auto x20 = x5*x8; 
      const auto x21 = x*x1; 
      const auto x22 = z*z*z*z; 
      const auto x23 = y*y*y*y*y; 
      const auto x24 = x11*x8; 
      const auto x25 = x15*x5; 
      const auto x26 = z*z*z*z*z; 
      const auto x27 = x*x*x*x*x*x; 
      const auto x28 = 5.0*radial_eval; 
      const auto x29 = 4.0*radial_eval; 
      const auto x30 = radial_eval_alpha*x0 + x29*x4; 
      const auto x31 = x2*x5; 
      const auto x32 = radial_eval_alpha*x31; 
      const auto x33 = 3.0*radial_eval; 
      const auto x34 = x13*x33; 
      const auto x35 = y*z; 
      const auto x36 = radial_eval_alpha*x2; 
      const auto x37 = x10*x33 + x36; 
      const auto x38 = x2*x8; 
      const auto x39 = radial_eval_alpha*x38; 
      const auto x40 = x14*x33; 
      const auto x41 = 2.0*x17; 
      const auto x42 = x11*x4; 
      const auto x43 = radial_eval_alpha*x42; 
      const auto x44 = radial_eval_alpha*x6; 
      const auto x45 = x41*x5 + x44; 
      const auto x46 = radial_eval_alpha*x9; 
      const auto x47 = x41*x8 + x46; 
      const auto x48 = x15*x4; 
      const auto x49 = radial_eval_alpha*x48; 
      const auto x50 = radial_eval*x18; 
      const auto x51 = x10*x18; 
      const auto x52 = radial_eval_alpha*x51; 
      const auto x53 = radial_eval*x11; 
      const auto x54 = radial_eval_alpha*x12; 
      const auto x55 = x53 + x54; 
      const auto x56 = radial_eval_alpha*x10*x5*x8; 
      const auto x57 = radial_eval*x15; 
      const auto x58 = radial_eval_alpha*x16; 
      const auto x59 = x57 + x58; 
      const auto x60 = radial_eval*x22; 
      const auto x61 = x10*x22; 
      const auto x62 = radial_eval_alpha*x61; 
      const auto x63 = radial_eval_alpha*x; 
      const auto x64 = x18*x63*z; 
      const auto x65 = x22*x63*y; 
      const auto x66 = radial_eval_alpha*y; 
      const auto x67 = radial_eval*x2; 
      const auto x68 = radial_eval_alpha*x2*x35; 
      const auto x69 = 2.0*x1; 
      const auto x70 = radial_eval*x4; 
      const auto x71 = x44 + x70; 
      const auto x72 = x10*x69 + x54; 
      const auto x73 = radial_eval_alpha*x23 + x11*x29; 
      const auto x74 = x*z; 
      const auto x75 = radial_eval_alpha*x18; 
      const auto x76 = x33*x5 + x75; 
      const auto x77 = radial_eval_alpha*x24; 
      const auto x78 = x69*x8 + x77; 
      const auto x79 = radial_eval_alpha*x25; 
      const auto x80 = y*y*y*y*y*y; 
      const auto x81 = x18*x8; 
      const auto x82 = radial_eval_alpha*x81; 
      const auto x83 = x20*x33; 
      const auto x84 = x11*x15; 
      const auto x85 = radial_eval_alpha*x84; 
      const auto x86 = x22*x5; 
      const auto x87 = radial_eval_alpha*x86; 
      const auto x88 = radial_eval_alpha*z; 
      const auto x89 = 2.0*x3; 
      const auto x90 = x10*x89 + x58; 
      const auto x91 = x5*x89 + x79; 
      const auto x92 = x*y; 
      const auto x93 = radial_eval_alpha*x22; 
      const auto x94 = x33*x8 + x93; 
      const auto x95 = radial_eval_alpha*x26 + x15*x29; 
      const auto x96 = z*z*z*z*z*z; 
      const auto x97 = 20.0*radial_eval; 
      const auto x98 = 10.0*radial_eval_alpha; 
      const auto x99 = radial_eval_alpha + radial_eval_alpha_squared*x10; 
      const auto x100 = x0*x98 + x0*x99 + x4*x97; 
      const auto x101 = 8.0*radial_eval_alpha; 
      const auto x102 = 12.0*radial_eval; 
      const auto x103 = x10*x102 + x2*x99; 
      const auto x104 = x101*x2 + x103; 
      const auto x105 = 6.0*radial_eval_alpha; 
      const auto x106 = 6.0*x17; 
      const auto x107 = x106*x5; 
      const auto x108 = x107 + x4*x5*x99; 
      const auto x109 = x106 + x4*x99; 
      const auto x110 = x106*x8; 
      const auto x111 = x110 + x4*x8*x99; 
      const auto x112 = 4.0*radial_eval_alpha; 
      const auto x113 = x112*x12; 
      const auto x114 = 2.0*radial_eval; 
      const auto x115 = x11*x114; 
      const auto x116 = x10*x11*x99 + x115; 
      const auto x117 = x112*x13; 
      const auto x118 = x114*x5; 
      const auto x119 = x10*x5*x99 + x118; 
      const auto x120 = x112*x14; 
      const auto x121 = x114*x8; 
      const auto x122 = x10*x8*x99 + x121; 
      const auto x123 = x112*x16; 
      const auto x124 = x114*x15; 
      const auto x125 = x10*x15*x99 + x124; 
      const auto x126 = 2.0*radial_eval_alpha; 
      const auto x127 = x126*x18; 
      const auto x128 = x18*x99; 
      const auto x129 = x11*x126; 
      const auto x130 = x11*x99; 
      const auto x131 = x5*x8*x99; 
      const auto x132 = x126*x15; 
      const auto x133 = x15*x99; 
      const auto x134 = x126*x22; 
      const auto x135 = x22*x99; 
      const auto x136 = x23*x99; 
      const auto x137 = x11*x8*x99; 
      const auto x138 = x15*x5*x99; 
      const auto x139 = x26*x99; 
      const auto x140 = 5.0*radial_eval_alpha; 
      const auto x141 = radial_eval_alpha_squared*x27 + x140*x2; 
      const auto x142 = x112*x6; 
      const auto x143 = x35*(radial_eval_alpha_squared*x0 + x112*x4); 
      const auto x144 = 6.0*x1; 
      const auto x145 = x10*x144; 
      const auto x146 = 2.0*x66; 
      const auto x147 = 3.0*radial_eval_alpha; 
      const auto x148 = radial_eval_alpha_squared*x31; 
      const auto x149 = x13*x147; 
      const auto x150 = x148 + x149; 
      const auto x151 = radial_eval_alpha_squared*x38; 
      const auto x152 = x14*x147; 
      const auto x153 = x151 + x152; 
      const auto x154 = 2.0*x63; 
      const auto x155 = x146*x4; 
      const auto x156 = radial_eval_alpha_squared*x42; 
      const auto x157 = x11*x154 + x156; 
      const auto x158 = radial_eval_alpha_squared*x4*x5*x8; 
      const auto x159 = x154*x20 + x158; 
      const auto x160 = radial_eval_alpha_squared*x48; 
      const auto x161 = x15*x154 + x160; 
      const auto x162 = radial_eval_alpha_squared*x51; 
      const auto x163 = x149 + x162; 
      const auto x164 = radial_eval_alpha_squared*x10*x11*x8; 
      const auto x165 = x14*x146 + x164; 
      const auto x166 = radial_eval_alpha_squared*x10*x15*x5; 
      const auto x167 = radial_eval_alpha_squared*x61; 
      const auto x168 = radial_eval_alpha_squared*x80 + x140*x18; 
      const auto x169 = x74*(radial_eval_alpha_squared*x23 + x11*x112); 
      const auto x170 = radial_eval_alpha_squared*x81; 
      const auto x171 = x147*x20; 
      const auto x172 = x170 + x171; 
      const auto x173 = radial_eval_alpha_squared*x84; 
      const auto x174 = x146*x15 + x173; 
      const auto x175 = radial_eval_alpha_squared*x86; 
      const auto x176 = x112*x9; 
      const auto x177 = 6.0*x3; 
      const auto x178 = x10*x177; 
      const auto x179 = 2.0*x88; 
      const auto x180 = x179*x4; 
      const auto x181 = x13*x179 + x166; 
      const auto x182 = x152 + x167; 
      const auto x183 = x11*x179; 
      const auto x184 = x171 + x175; 
      const auto x185 = x92*(radial_eval_alpha_squared*x26 + x112*x15); 
      const auto x186 = radial_eval_alpha_squared*x96 + x140*x22; 
      const auto x187 = radial_eval_alpha + radial_eval_alpha_squared*x5; 
      const auto x188 = x0*x187; 
      const auto x189 = x126*x2; 
      const auto x190 = x187*x2; 
      const auto x191 = x114*x4; 
      const auto x192 = x187*x4*x5 + x191; 
      const auto x193 = x126*x4; 
      const auto x194 = x187*x4; 
      const auto x195 = x187*x4*x8; 
      const auto x196 = x10*x11*x187 + x145; 
      const auto x197 = x10*x114; 
      const auto x198 = x10*x187*x5 + x197; 
      const auto x199 = x10*x187*x8; 
      const auto x200 = x10*x15*x187; 
      const auto x201 = x102*x5 + x18*x187; 
      const auto x202 = x101*x18 + x201; 
      const auto x203 = x11*x187 + x144; 
      const auto x204 = x112*x20; 
      const auto x205 = x121 + x187*x5*x8; 
      const auto x206 = x15*x187; 
      const auto x207 = x187*x22; 
      const auto x208 = x11*x97 + x187*x23 + x23*x98; 
      const auto x209 = x144*x8; 
      const auto x210 = x11*x187*x8 + x209; 
      const auto x211 = x112*x25; 
      const auto x212 = x124 + x15*x187*x5; 
      const auto x213 = x187*x26; 
      const auto x214 = x112*x24; 
      const auto x215 = x177*x5; 
      const auto x216 = radial_eval_alpha + radial_eval_alpha_squared*x8; 
      const auto x217 = x0*x216; 
      const auto x218 = x2*x216; 
      const auto x219 = x216*x4*x5; 
      const auto x220 = x216*x4; 
      const auto x221 = x191 + x216*x4*x8; 
      const auto x222 = x10*x11*x216; 
      const auto x223 = x10*x216*x5; 
      const auto x224 = x10*x216*x8 + x197; 
      const auto x225 = x10*x15*x216 + x178; 
      const auto x226 = x18*x216; 
      const auto x227 = x11*x216; 
      const auto x228 = x118 + x216*x5*x8; 
      const auto x229 = x15*x216 + x177; 
      const auto x230 = x102*x8 + x216*x22; 
      const auto x231 = x101*x22 + x230; 
      const auto x232 = x216*x23; 
      const auto x233 = x11*x216*x8 + x115; 
      const auto x234 = x15*x216*x5 + x215; 
      const auto x235 = x15*x97 + x216*x26 + x26*x98; 
      const auto x236 = x103 + x190 + x2*x98 + x218; 
      const auto x237 = x128 + x18*x98 + x201 + x226; 
      const auto x238 = x135 + x207 + x22*x98 + x230; 


      // Evaluate basis function
      basis_eval[ipt + 0*npts] = radial_eval*x0;
      basis_eval[ipt + 1*npts] = x1*x2;
      basis_eval[ipt + 2*npts] = x2*x3;
      basis_eval[ipt + 3*npts] = radial_eval*x6;
      basis_eval[ipt + 4*npts] = x4*x7;
      basis_eval[ipt + 5*npts] = radial_eval*x9;
      basis_eval[ipt + 6*npts] = radial_eval*x12;
      basis_eval[ipt + 7*npts] = x13*x3;
      basis_eval[ipt + 8*npts] = x1*x14;
      basis_eval[ipt + 9*npts] = radial_eval*x16;
      basis_eval[ipt + 10*npts] = x17*x18;
      basis_eval[ipt + 11*npts] = x11*x19;
      basis_eval[ipt + 12*npts] = x17*x20;
      basis_eval[ipt + 13*npts] = x15*x21;
      basis_eval[ipt + 14*npts] = x17*x22;
      basis_eval[ipt + 15*npts] = radial_eval*x23;
      basis_eval[ipt + 16*npts] = x18*x3;
      basis_eval[ipt + 17*npts] = radial_eval*x24;
      basis_eval[ipt + 18*npts] = radial_eval*x25;
      basis_eval[ipt + 19*npts] = x1*x22;
      basis_eval[ipt + 20*npts] = radial_eval*x26;


    
      // Evaluate first derivative of bfn wrt x
      basis_x_eval[ipt + 0*npts] = radial_eval_alpha*x27 + x2*x28;
      basis_x_eval[ipt + 1*npts] = x30*y;
      basis_x_eval[ipt + 2*npts] = x30*z;
      basis_x_eval[ipt + 3*npts] = x32 + x34;
      basis_x_eval[ipt + 4*npts] = x35*x37;
      basis_x_eval[ipt + 5*npts] = x39 + x40;
      basis_x_eval[ipt + 6*npts] = x11*x41 + x43;
      basis_x_eval[ipt + 7*npts] = x45*z;
      basis_x_eval[ipt + 8*npts] = x47*y;
      basis_x_eval[ipt + 9*npts] = x15*x41 + x49;
      basis_x_eval[ipt + 10*npts] = x50 + x52;
      basis_x_eval[ipt + 11*npts] = x55*z;
      basis_x_eval[ipt + 12*npts] = radial_eval*x20 + x56;
      basis_x_eval[ipt + 13*npts] = x59*y;
      basis_x_eval[ipt + 14*npts] = x60 + x62;
      basis_x_eval[ipt + 15*npts] = x23*x63;
      basis_x_eval[ipt + 16*npts] = x64;
      basis_x_eval[ipt + 17*npts] = x24*x63;
      basis_x_eval[ipt + 18*npts] = x25*x63;
      basis_x_eval[ipt + 19*npts] = x65;
      basis_x_eval[ipt + 20*npts] = x26*x63;

      // Evaluate first derivative of bfn wrt y
      basis_y_eval[ipt + 0*npts] = x0*x66;
      basis_y_eval[ipt + 1*npts] = x32 + x67;
      basis_y_eval[ipt + 2*npts] = x68;
      basis_y_eval[ipt + 3*npts] = x4*x69 + x43;
      basis_y_eval[ipt + 4*npts] = x71*z;
      basis_y_eval[ipt + 5*npts] = x66*x9;
      basis_y_eval[ipt + 6*npts] = x34 + x52;
      basis_y_eval[ipt + 7*npts] = x72*z;
      basis_y_eval[ipt + 8*npts] = radial_eval*x14 + x56;
      basis_y_eval[ipt + 9*npts] = x16*x66;
      basis_y_eval[ipt + 10*npts] = x*x73;
      basis_y_eval[ipt + 11*npts] = x74*x76;
      basis_y_eval[ipt + 12*npts] = x*x78;
      basis_y_eval[ipt + 13*npts] = x*(x57 + x79);
      basis_y_eval[ipt + 14*npts] = x65;
      basis_y_eval[ipt + 15*npts] = radial_eval_alpha*x80 + x18*x28;
      basis_y_eval[ipt + 16*npts] = x73*z;
      basis_y_eval[ipt + 17*npts] = x82 + x83;
      basis_y_eval[ipt + 18*npts] = x15*x69 + x85;
      basis_y_eval[ipt + 19*npts] = x60 + x87;
      basis_y_eval[ipt + 20*npts] = x26*x66;

      // Evaluate first derivative of bfn wrt z
      basis_z_eval[ipt + 0*npts] = x0*x88;
      basis_z_eval[ipt + 1*npts] = x68;
      basis_z_eval[ipt + 2*npts] = x39 + x67;
      basis_z_eval[ipt + 3*npts] = x6*x88;
      basis_z_eval[ipt + 4*npts] = y*(x46 + x70);
      basis_z_eval[ipt + 5*npts] = x4*x89 + x49;
      basis_z_eval[ipt + 6*npts] = x12*x88;
      basis_z_eval[ipt + 7*npts] = radial_eval*x13 + x56;
      basis_z_eval[ipt + 8*npts] = x90*y;
      basis_z_eval[ipt + 9*npts] = x40 + x62;
      basis_z_eval[ipt + 10*npts] = x64;
      basis_z_eval[ipt + 11*npts] = x*(x53 + x77);
      basis_z_eval[ipt + 12*npts] = x*x91;
      basis_z_eval[ipt + 13*npts] = x92*x94;
      basis_z_eval[ipt + 14*npts] = x*x95;
      basis_z_eval[ipt + 15*npts] = x23*x88;
      basis_z_eval[ipt + 16*npts] = x50 + x82;
      basis_z_eval[ipt + 17*npts] = x11*x89 + x85;
      basis_z_eval[ipt + 18*npts] = x83 + x87;
      basis_z_eval[ipt + 19*npts] = x95*y;
      basis_z_eval[ipt + 20*npts] = radial_eval_alpha*x96 + x22*x28;


      // Evaluate Laplacian of bfn 
      basis_lapl_eval[ipt + 0*npts] = x100 + x188 + x217;
      basis_lapl_eval[ipt + 1*npts] = x236*y;
      basis_lapl_eval[ipt + 2*npts] = x236*z;
      basis_lapl_eval[ipt + 3*npts] = x108 + x192 + x219 + x6*x98;
      basis_lapl_eval[ipt + 4*npts] = x35*(x109 + x194 + x220 + x4*x98);
      basis_lapl_eval[ipt + 5*npts] = x111 + x195 + x221 + x9*x98;
      basis_lapl_eval[ipt + 6*npts] = x116 + x12*x98 + x196 + x222;
      basis_lapl_eval[ipt + 7*npts] = z*(x119 + x13*x98 + x198 + x223);
      basis_lapl_eval[ipt + 8*npts] = y*(x122 + x14*x98 + x199 + x224);
      basis_lapl_eval[ipt + 9*npts] = x125 + x16*x98 + x200 + x225;
      basis_lapl_eval[ipt + 10*npts] = x*x237;
      basis_lapl_eval[ipt + 11*npts] = x74*(x11*x98 + x130 + x203 + x227);
      basis_lapl_eval[ipt + 12*npts] = x*(x131 + x20*x98 + x205 + x228);
      basis_lapl_eval[ipt + 13*npts] = x92*(x133 + x15*x98 + x206 + x229);
      basis_lapl_eval[ipt + 14*npts] = x*x238;
      basis_lapl_eval[ipt + 15*npts] = x136 + x208 + x232;
      basis_lapl_eval[ipt + 16*npts] = x237*z;
      basis_lapl_eval[ipt + 17*npts] = x137 + x210 + x233 + x24*x98;
      basis_lapl_eval[ipt + 18*npts] = x138 + x212 + x234 + x25*x98;
      basis_lapl_eval[ipt + 19*npts] = x238*y;
      basis_lapl_eval[ipt + 20*npts] = x139 + x213 + x235;





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

      ang_eval_0 = x4*x7;
      ang_eval_1 = radial_eval*x9;
      ang_eval_2 = radial_eval*x12;
      ang_eval_3 = x13*x3;
      basis_eval[ipt + 4*npts] = ang_eval_0;
      basis_eval[ipt + 5*npts] = ang_eval_1;
      basis_eval[ipt + 6*npts] = ang_eval_2;
      basis_eval[ipt + 7*npts] = ang_eval_3;

      ang_eval_0 = x1*x14;
      ang_eval_1 = radial_eval*x16;
      ang_eval_2 = x17*x18;
      ang_eval_3 = x11*x19;
      basis_eval[ipt + 8*npts] = ang_eval_0;
      basis_eval[ipt + 9*npts] = ang_eval_1;
      basis_eval[ipt + 10*npts] = ang_eval_2;
      basis_eval[ipt + 11*npts] = ang_eval_3;

      ang_eval_0 = x17*x20;
      ang_eval_1 = x15*x21;
      ang_eval_2 = x17*x22;
      ang_eval_3 = radial_eval*x23;
      basis_eval[ipt + 12*npts] = ang_eval_0;
      basis_eval[ipt + 13*npts] = ang_eval_1;
      basis_eval[ipt + 14*npts] = ang_eval_2;
      basis_eval[ipt + 15*npts] = ang_eval_3;

      ang_eval_0 = x18*x3;
      ang_eval_1 = radial_eval*x24;
      ang_eval_2 = radial_eval*x25;
      ang_eval_3 = x1*x22;
      basis_eval[ipt + 16*npts] = ang_eval_0;
      basis_eval[ipt + 17*npts] = ang_eval_1;
      basis_eval[ipt + 18*npts] = ang_eval_2;
      basis_eval[ipt + 19*npts] = ang_eval_3;

      ang_eval_0 = radial_eval*x26;
      basis_eval[ipt + 20*npts] = ang_eval_0;


      double dang_eval_x_0, dang_eval_y_0, dang_eval_z_0;
      double dang_eval_x_1, dang_eval_y_1, dang_eval_z_1;
      double dang_eval_x_2, dang_eval_y_2, dang_eval_z_2;
      double dang_eval_x_3, dang_eval_y_3, dang_eval_z_3;

      dang_eval_x_0 = radial_eval_alpha*x27 + x2*x28;
      dang_eval_y_0 = x0*x66;
      dang_eval_z_0 = x0*x88;
      dang_eval_x_1 = x30*y;
      dang_eval_y_1 = x32 + x67;
      dang_eval_z_1 = x68;
      dang_eval_x_2 = x30*z;
      dang_eval_y_2 = x68;
      dang_eval_z_2 = x39 + x67;
      dang_eval_x_3 = x32 + x34;
      dang_eval_y_3 = x4*x69 + x43;
      dang_eval_z_3 = x6*x88;
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

      dang_eval_x_0 = x35*x37;
      dang_eval_y_0 = x71*z;
      dang_eval_z_0 = y*(x46 + x70);
      dang_eval_x_1 = x39 + x40;
      dang_eval_y_1 = x66*x9;
      dang_eval_z_1 = x4*x89 + x49;
      dang_eval_x_2 = x11*x41 + x43;
      dang_eval_y_2 = x34 + x52;
      dang_eval_z_2 = x12*x88;
      dang_eval_x_3 = x45*z;
      dang_eval_y_3 = x72*z;
      dang_eval_z_3 = radial_eval*x13 + x56;
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

      dang_eval_x_0 = x47*y;
      dang_eval_y_0 = radial_eval*x14 + x56;
      dang_eval_z_0 = x90*y;
      dang_eval_x_1 = x15*x41 + x49;
      dang_eval_y_1 = x16*x66;
      dang_eval_z_1 = x40 + x62;
      dang_eval_x_2 = x50 + x52;
      dang_eval_y_2 = x*x73;
      dang_eval_z_2 = x64;
      dang_eval_x_3 = x55*z;
      dang_eval_y_3 = x74*x76;
      dang_eval_z_3 = x*(x53 + x77);
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

      dang_eval_x_0 = radial_eval*x20 + x56;
      dang_eval_y_0 = x*x78;
      dang_eval_z_0 = x*x91;
      dang_eval_x_1 = x59*y;
      dang_eval_y_1 = x*(x57 + x79);
      dang_eval_z_1 = x92*x94;
      dang_eval_x_2 = x60 + x62;
      dang_eval_y_2 = x65;
      dang_eval_z_2 = x*x95;
      dang_eval_x_3 = x23*x63;
      dang_eval_y_3 = radial_eval_alpha*x80 + x18*x28;
      dang_eval_z_3 = x23*x88;
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

      dang_eval_x_0 = x64;
      dang_eval_y_0 = x73*z;
      dang_eval_z_0 = x50 + x82;
      dang_eval_x_1 = x24*x63;
      dang_eval_y_1 = x82 + x83;
      dang_eval_z_1 = x11*x89 + x85;
      dang_eval_x_2 = x25*x63;
      dang_eval_y_2 = x15*x69 + x85;
      dang_eval_z_2 = x83 + x87;
      dang_eval_x_3 = x65;
      dang_eval_y_3 = x60 + x87;
      dang_eval_z_3 = x95*y;
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

      dang_eval_x_0 = x26*x63;
      dang_eval_y_0 = x26*x66;
      dang_eval_z_0 = radial_eval_alpha*x96 + x22*x28;
      basis_x_eval[ipt + 20*npts] = dang_eval_x_0;
      basis_y_eval[ipt + 20*npts] = dang_eval_y_0;
      basis_z_eval[ipt + 20*npts] = dang_eval_z_0;

#endif
    } // Loop over points within task
  } // Loop over tasks
        
  } // Loop over shells
} // end kernel

} // namespace GauXC
