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


__global__ __launch_bounds__(128,2) void collocation_device_shell_to_task_kernel_spherical_laplacian_5(
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
      const auto x0 = 0.1875*sqrt_14; 
      const auto x1 = x0*y; 
      const auto x2 = y*y*y*y; 
      const auto x3 = x*x*x*x; 
      const auto x4 = x*x; 
      const auto x5 = x4; 
      const auto x6 = y*y; 
      const auto x7 = x6; 
      const auto x8 = x5*x7; 
      const auto x9 = -10.0*x8; 
      const auto x10 = x2 + 5.0*x3 + x9; 
      const auto x11 = 1.5*sqrt_35; 
      const auto x12 = x11*z; 
      const auto x13 = x12*y; 
      const auto x14 = radial_eval*x; 
      const auto x15 = -x7; 
      const auto x16 = x15 + x5; 
      const auto x17 = 0.0625*sqrt_70; 
      const auto x18 = x17*y; 
      const auto x19 = 3.0*x3; 
      const auto x20 = 2.0*x8; 
      const auto x21 = z*z; 
      const auto x22 = x21; 
      const auto x23 = x22*x7; 
      const auto x24 = 8.0*x23; 
      const auto x25 = -x19 + x2 - x20 + 24.0*x22*x5 - x24; 
      const auto x26 = 0.5*sqrt_105; 
      const auto x27 = x26*z; 
      const auto x28 = x27*y; 
      const auto x29 = -2.0*x22; 
      const auto x30 = x29 + x7; 
      const auto x31 = -x30 - x5; 
      const auto x32 = 0.125*sqrt_15; 
      const auto x33 = x32*y; 
      const auto x34 = 8.0*(z*z*z*z); 
      const auto x35 = x22*x5; 
      const auto x36 = x2 + x3; 
      const auto x37 = x20 - 12.0*x23 + x34 - 12.0*x35 + x36; 
      const auto x38 = 0.125*z; 
      const auto x39 = 15.0*x2 - 40.0*x23 + 15.0*x3 + x34 - 40.0*x35 + 30.0*x8; 
      const auto x40 = x*x32; 
      const auto x41 = 0.25*sqrt_105; 
      const auto x42 = x41*z; 
      const auto x43 = -x3; 
      const auto x44 = x2 - 2.0*x23 + 2.0*x35 + x43; 
      const auto x45 = x*x17; 
      const auto x46 = 3.0*x2 + x20 - 24.0*x23 + 8.0*x35 + x43; 
      const auto x47 = 0.375*sqrt_35; 
      const auto x48 = x47*z; 
      const auto x49 = x36 - 6.0*x8; 
      const auto x50 = x*x0; 
      const auto x51 = 5.0*x2 + x3 + x9; 
      const auto x52 = 20.0*radial_eval; 
      const auto x53 = x*x*x; 
      const auto x54 = x*x7; 
      const auto x55 = x53 - x54; 
      const auto x56 = radial_eval_alpha*x; 
      const auto x57 = 3.0*x5; 
      const auto x58 = x15 + x57; 
      const auto x59 = radial_eval*x58 + radial_eval_alpha*x16*x5; 
      const auto x60 = 4.0*radial_eval; 
      const auto x61 = 3.0*x53; 
      const auto x62 = -12.0*x*x22 + x61; 
      const auto x63 = -x54 - x62; 
      const auto x64 = -x30 - x57; 
      const auto x65 = -6.0*x*x22 + x53; 
      const auto x66 = x54 + x65; 
      const auto x67 = 3.0*x; 
      const auto x68 = x67*x7; 
      const auto x69 = 4.0*x; 
      const auto x70 = x22*x69; 
      const auto x71 = -x70; 
      const auto x72 = x61 + x68 + x71; 
      const auto x73 = x37*x5; 
      const auto x74 = x37 + x66*x69; 
      const auto x75 = -x53; 
      const auto x76 = x*x22 + x75; 
      const auto x77 = x46*x5; 
      const auto x78 = x54 + x70 + x75; 
      const auto x79 = x46 + x69*x78; 
      const auto x80 = x53 - x68; 
      const auto x81 = x49*x56 + x60*x80; 
      const auto x82 = x5*x51; 
      const auto x83 = -5.0*x*x7 + x53; 
      const auto x84 = x51 + x69*x83; 
      const auto x85 = x10*x7; 
      const auto x86 = 4.0*y; 
      const auto x87 = y*y*y; 
      const auto x88 = -5.0*x5*y + x87; 
      const auto x89 = x10 + x86*x88; 
      const auto x90 = 3.0*x7; 
      const auto x91 = -x90; 
      const auto x92 = x5 + x91; 
      const auto x93 = radial_eval*x92 + radial_eval_alpha*x16*x7; 
      const auto x94 = x25*x7; 
      const auto x95 = x5*y; 
      const auto x96 = x22*x86; 
      const auto x97 = x87 - x95 - x96; 
      const auto x98 = -x19 + x2 - x20 - x24 + 24.0*x35 + x86*x97; 
      const auto x99 = x29 + x90; 
      const auto x100 = -x5 - x99; 
      const auto x101 = x37*x7; 
      const auto x102 = -6.0*x22*y + x87; 
      const auto x103 = x102 + x95; 
      const auto x104 = x103*x86 + x37; 
      const auto x105 = -x96; 
      const auto x106 = 3.0*x87; 
      const auto x107 = 3.0*y; 
      const auto x108 = x107*x5; 
      const auto x109 = x106 + x108; 
      const auto x110 = x105 + x109; 
      const auto x111 = radial_eval_alpha*y; 
      const auto x112 = -x22*y + x87; 
      const auto x113 = -12.0*x22*y; 
      const auto x114 = x106 + x113 + x95; 
      const auto x115 = -x108 + x87; 
      const auto x116 = x111*x49 + x115*x60; 
      const auto x117 = x87 - x95; 
      const auto x118 = x0*z; 
      const auto x119 = x11*y; 
      const auto x120 = x18*z; 
      const auto x121 = 16.0*radial_eval; 
      const auto x122 = x121*x58; 
      const auto x123 = x26*y; 
      const auto x124 = -6.0*x22; 
      const auto x125 = x124 + x7; 
      const auto x126 = -x125 - x5; 
      const auto x127 = 8.0*radial_eval; 
      const auto x128 = 3.0*z; 
      const auto x129 = x128*x5; 
      const auto x130 = z*z*z; 
      const auto x131 = -4.0*x130; 
      const auto x132 = x128*x7 + x131; 
      const auto x133 = -x129 - x132; 
      const auto x134 = radial_eval_alpha*z; 
      const auto x135 = x127*x133 + x134*x37; 
      const auto x136 = x22*x39; 
      const auto x137 = 16.0*z; 
      const auto x138 = 5.0*z; 
      const auto x139 = -x137*(-2.0*x130 + x138*x5 + x138*x7) + x39; 
      const auto x140 = x22*x44; 
      const auto x141 = 4.0*x16*x21 + x44; 
      const auto x142 = x45*z; 
      const auto x143 = x121*x92; 
      const auto x144 = x22*x49; 
      const auto x145 = 40.0*x56; 
      const auto x146 = radial_eval_alpha + radial_eval_alpha_squared*x5; 
      const auto x147 = x10*x146; 
      const auto x148 = x*x13; 
      const auto x149 = 6.0*radial_eval; 
      const auto x150 = 2.0*radial_eval_alpha; 
      const auto x151 = x146*x16 + x150*x58; 
      const auto x152 = 9.0*x5; 
      const auto x153 = -12.0*x22; 
      const auto x154 = -x152 - x153 - x7; 
      const auto x155 = 8.0*x56; 
      const auto x156 = x146*x25; 
      const auto x157 = x*x28; 
      const auto x158 = -x149; 
      const auto x159 = x146*x31 + x150*x64; 
      const auto x160 = x125 + x57; 
      const auto x161 = x146*x37; 
      const auto x162 = 4.0*x22; 
      const auto x163 = -x162; 
      const auto x164 = x152 + x163 + x90; 
      const auto x165 = x146*x39; 
      const auto x166 = 2.0*x56; 
      const auto x167 = x*x161 + x166*x74 + x60*(x*x160 + 2.0*x66); 
      const auto x168 = -x57; 
      const auto x169 = x146*x44 + x155*x76 + x60*(x168 + x22); 
      const auto x170 = x*x146*x46 + x166*x79 + x60*(x*(x162 + x168 + x7) + 2.0*x78); 
      const auto x171 = 12.0*radial_eval; 
      const auto x172 = x146*x49 + x155*x80 + x16*x171; 
      const auto x173 = 5.0*x7; 
      const auto x174 = x*x146*x51 + x166*x84 - x60*(x*(x173 - x57) - 2.0*x83); 
      const auto x175 = 20.0*radial_eval_alpha; 
      const auto x176 = radial_eval_alpha_squared*x; 
      const auto x177 = x57 + x91; 
      const auto x178 = radial_eval_alpha*x4; 
      const auto x179 = radial_eval_alpha*x6; 
      const auto x180 = 4.0*radial_eval_alpha; 
      const auto x181 = x176*y; 
      const auto x182 = radial_eval_alpha_squared*y; 
      const auto x183 = 4.0*x56; 
      const auto x184 = radial_eval_alpha*x86; 
      const auto x185 = 24.0*radial_eval; 
      const auto x186 = x185*y; 
      const auto x187 = radial_eval_alpha*x21; 
      const auto x188 = 96.0*radial_eval; 
      const auto x189 = 48.0*z; 
      const auto x190 = 4.0*x134; 
      const auto x191 = 60.0*radial_eval; 
      const auto x192 = 9.0*z; 
      const auto x193 = 8.0*radial_eval_alpha; 
      const auto x194 = radial_eval_alpha_squared*z; 
      const auto x195 = x17*z; 
      const auto x196 = x121*x177; 
      const auto x197 = 16.0*radial_eval_alpha; 
      const auto x198 = radial_eval_alpha + radial_eval_alpha_squared*x7; 
      const auto x199 = 5.0*x5; 
      const auto x200 = 2.0*x111; 
      const auto x201 = x10*x198*y + x200*x89 + x60*(2.0*x88 - y*(x199 - x90)); 
      const auto x202 = x150*x92 + x16*x198; 
      const auto x203 = x198*x25*y + x200*x98 + x60*(2.0*x97 - y*(x162 + x92)); 
      const auto x204 = x100*x150 + x198*x31; 
      const auto x205 = x124 + x5 + x90; 
      const auto x206 = x198*x37; 
      const auto x207 = x104*x200 + x206*y + x60*(2.0*x103 + x205*y); 
      const auto x208 = 9.0*x7; 
      const auto x209 = x163 + x208 + x57; 
      const auto x210 = 40.0*x111; 
      const auto x211 = x198*x39; 
      const auto x212 = 8.0*x111; 
      const auto x213 = x112*x212 + x198*x44 - x60*(x22 - x90); 
      const auto x214 = x153 + x208 + x5; 
      const auto x215 = x198*x46; 
      const auto x216 = -x5; 
      const auto x217 = x115*x212 + x171*(x216 + x7) + x198*x49; 
      const auto x218 = x216 + x90; 
      const auto x219 = x198*x51; 
      const auto x220 = radial_eval_alpha + radial_eval_alpha_squared*x22; 
      const auto x221 = x10*x220; 
      const auto x222 = x16*(x150 + x220); 
      const auto x223 = 32.0*x187; 
      const auto x224 = x220*x25; 
      const auto x225 = x126*x150 + x220*x31; 
      const auto x226 = -x163 - x5 - x7; 
      const auto x227 = x220*x37; 
      const auto x228 = 16.0*x133*x134 + x185*x226 + x227; 
      const auto x229 = 10.0*z; 
      const auto x230 = -x121*(-4.0*x130 + x229*x5 + x229*x7 + z*(x124 + x173 + x199)) + 2.0*x134*x139 + x220*x39*z; 
      const auto x231 = x141*x150 + 3.0*x16*x60 + x220*x44; 
      const auto x232 = x220*x46; 
      const auto x233 = x49*(x150 + x220); 
      const auto x234 = x220*x51; 
      const auto x235 = x145*y; 
      const auto x236 = x60*y; 
      const auto x237 = x155*y; 
      const auto x238 = 32.0*x21; 
      const auto x239 = x52*z; 
      const auto x240 = x*x60; 


      // Evaluate basis function
      basis_eval[ipt + 0*npts] = radial_eval*x1*x10;
      basis_eval[ipt + 1*npts] = x13*x14*x16;
      basis_eval[ipt + 2*npts] = radial_eval*x18*x25;
      basis_eval[ipt + 3*npts] = x14*x28*x31;
      basis_eval[ipt + 4*npts] = radial_eval*x33*x37;
      basis_eval[ipt + 5*npts] = radial_eval*x38*x39;
      basis_eval[ipt + 6*npts] = radial_eval*x37*x40;
      basis_eval[ipt + 7*npts] = radial_eval*x42*x44;
      basis_eval[ipt + 8*npts] = radial_eval*x45*x46;
      basis_eval[ipt + 9*npts] = radial_eval*x48*x49;
      basis_eval[ipt + 10*npts] = radial_eval*x50*x51;


    
      // Evaluate first derivative of bfn wrt x
      basis_x_eval[ipt + 0*npts] = x1*(x10*x56 + x52*x55);
      basis_x_eval[ipt + 1*npts] = x13*x59;
      basis_x_eval[ipt + 2*npts] = x18*(x25*x56 + x60*x63);
      basis_x_eval[ipt + 3*npts] = x28*(radial_eval*x64 + radial_eval_alpha*x31*x5);
      basis_x_eval[ipt + 4*npts] = x33*(x37*x56 + x60*x66);
      basis_x_eval[ipt + 5*npts] = x38*(x39*x56 + x52*x72);
      basis_x_eval[ipt + 6*npts] = x32*(radial_eval*x74 + radial_eval_alpha*x73);
      basis_x_eval[ipt + 7*npts] = x42*(x44*x56 + x60*x76);
      basis_x_eval[ipt + 8*npts] = x17*(radial_eval*x79 + radial_eval_alpha*x77);
      basis_x_eval[ipt + 9*npts] = x48*x81;
      basis_x_eval[ipt + 10*npts] = x0*(radial_eval*x84 + radial_eval_alpha*x82);

      // Evaluate first derivative of bfn wrt y
      basis_y_eval[ipt + 0*npts] = x0*(radial_eval*x89 + radial_eval_alpha*x85);
      basis_y_eval[ipt + 1*npts] = x*x12*x93;
      basis_y_eval[ipt + 2*npts] = x17*(radial_eval*x98 + radial_eval_alpha*x94);
      basis_y_eval[ipt + 3*npts] = x*x27*(radial_eval*x100 + radial_eval_alpha*x31*x7);
      basis_y_eval[ipt + 4*npts] = x32*(radial_eval*x104 + radial_eval_alpha*x101);
      basis_y_eval[ipt + 5*npts] = x38*(x110*x52 + x111*x39);
      basis_y_eval[ipt + 6*npts] = x40*(x103*x60 + x111*x37);
      basis_y_eval[ipt + 7*npts] = x42*(x111*x44 + x112*x60);
      basis_y_eval[ipt + 8*npts] = x45*(x111*x46 + x114*x60);
      basis_y_eval[ipt + 9*npts] = x116*x48;
      basis_y_eval[ipt + 10*npts] = x50*(x111*x51 + x117*x52);

      // Evaluate first derivative of bfn wrt z
      basis_z_eval[ipt + 0*npts] = x10*x111*x118;
      basis_z_eval[ipt + 1*npts] = x*x119*x16*(radial_eval + radial_eval_alpha*x22);
      basis_z_eval[ipt + 2*npts] = x120*(radial_eval_alpha*x25 + x122);
      basis_z_eval[ipt + 3*npts] = x*x123*(radial_eval*x126 + radial_eval_alpha*x22*x31);
      basis_z_eval[ipt + 4*npts] = x135*x33;
      basis_z_eval[ipt + 5*npts] = 0.125*radial_eval*x139 + 0.125*radial_eval_alpha*x136;
      basis_z_eval[ipt + 6*npts] = x135*x40;
      basis_z_eval[ipt + 7*npts] = x41*(radial_eval*x141 + radial_eval_alpha*x140);
      basis_z_eval[ipt + 8*npts] = x142*(radial_eval_alpha*x46 + x143);
      basis_z_eval[ipt + 9*npts] = x47*(radial_eval*x49 + radial_eval_alpha*x144);
      basis_z_eval[ipt + 10*npts] = x118*x51*x56;


      // Evaluate Laplacian of bfn 
      basis_lapl_eval[ipt + 0*npts] = x0*(x147*y + x201 + x221*y + x235*x55 + x52*x58*y);
      basis_lapl_eval[ipt + 1*npts] = x148*(x151 + x202 + x222);
      basis_lapl_eval[ipt + 2*npts] = x17*(x111*x238*x58 + x121*x58*y + x154*x236 + x156*y + x203 + x224*y + x237*x63);
      basis_lapl_eval[ipt + 3*npts] = x157*(x159 + x204 + x225);
      basis_lapl_eval[ipt + 4*npts] = x32*(x111*x133*x137 + x160*x236 + x161*y + x186*x226 + x207 + x227*y + x237*x66);
      basis_lapl_eval[ipt + 5*npts] = 0.125*x110*x210*z + 0.125*x145*x72*z + 0.125*x164*x239 + 0.125*x165*z + 0.125*x209*x239 + 0.125*x211*z + 0.125*x230;
      basis_lapl_eval[ipt + 6*npts] = x32*(x*x185*x226 + x*x206 + x*x227 + x103*x237 + x133*x137*x56 + x167 + x205*x240);
      basis_lapl_eval[ipt + 7*npts] = x42*(x169 + x213 + x231);
      basis_lapl_eval[ipt + 8*npts] = x17*(x*x121*x92 + x*x215 + x*x232 + x114*x237 + x170 + x214*x240 + x238*x56*x92);
      basis_lapl_eval[ipt + 9*npts] = x48*(x172 + x217 + x233);
      basis_lapl_eval[ipt + 10*npts] = x0*(x*x218*x52 + x*x219 + x*x234 + x117*x235 + x174);





#if 0
      // Evaluate the angular part of bfn



      double ang_eval_0;
      double ang_eval_1;
      double ang_eval_2;
      double ang_eval_3;


      ang_eval_0 = radial_eval*x1*x10;
      ang_eval_1 = x13*x14*x16;
      ang_eval_2 = radial_eval*x18*x25;
      ang_eval_3 = x14*x28*x31;
      basis_eval[ipt + 0*npts] = ang_eval_0;
      basis_eval[ipt + 1*npts] = ang_eval_1;
      basis_eval[ipt + 2*npts] = ang_eval_2;
      basis_eval[ipt + 3*npts] = ang_eval_3;

      ang_eval_0 = radial_eval*x33*x37;
      ang_eval_1 = radial_eval*x38*x39;
      ang_eval_2 = radial_eval*x37*x40;
      ang_eval_3 = radial_eval*x42*x44;
      basis_eval[ipt + 4*npts] = ang_eval_0;
      basis_eval[ipt + 5*npts] = ang_eval_1;
      basis_eval[ipt + 6*npts] = ang_eval_2;
      basis_eval[ipt + 7*npts] = ang_eval_3;

      ang_eval_0 = radial_eval*x45*x46;
      ang_eval_1 = radial_eval*x48*x49;
      ang_eval_2 = radial_eval*x50*x51;
      basis_eval[ipt + 8*npts] = ang_eval_0;
      basis_eval[ipt + 9*npts] = ang_eval_1;
      basis_eval[ipt + 10*npts] = ang_eval_2;


      double dang_eval_x_0, dang_eval_y_0, dang_eval_z_0;
      double dang_eval_x_1, dang_eval_y_1, dang_eval_z_1;
      double dang_eval_x_2, dang_eval_y_2, dang_eval_z_2;
      double dang_eval_x_3, dang_eval_y_3, dang_eval_z_3;

      dang_eval_x_0 = x1*(x10*x56 + x52*x55);
      dang_eval_y_0 = x0*(radial_eval*x89 + radial_eval_alpha*x85);
      dang_eval_z_0 = x10*x111*x118;
      dang_eval_x_1 = x13*x59;
      dang_eval_y_1 = x*x12*x93;
      dang_eval_z_1 = x*x119*x16*(radial_eval + radial_eval_alpha*x22);
      dang_eval_x_2 = x18*(x25*x56 + x60*x63);
      dang_eval_y_2 = x17*(radial_eval*x98 + radial_eval_alpha*x94);
      dang_eval_z_2 = x120*(radial_eval_alpha*x25 + x122);
      dang_eval_x_3 = x28*(radial_eval*x64 + radial_eval_alpha*x31*x5);
      dang_eval_y_3 = x*x27*(radial_eval*x100 + radial_eval_alpha*x31*x7);
      dang_eval_z_3 = x*x123*(radial_eval*x126 + radial_eval_alpha*x22*x31);
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

      dang_eval_x_0 = x33*(x37*x56 + x60*x66);
      dang_eval_y_0 = x32*(radial_eval*x104 + radial_eval_alpha*x101);
      dang_eval_z_0 = x135*x33;
      dang_eval_x_1 = x38*(x39*x56 + x52*x72);
      dang_eval_y_1 = x38*(x110*x52 + x111*x39);
      dang_eval_z_1 = 0.125*radial_eval*x139 + 0.125*radial_eval_alpha*x136;
      dang_eval_x_2 = x32*(radial_eval*x74 + radial_eval_alpha*x73);
      dang_eval_y_2 = x40*(x103*x60 + x111*x37);
      dang_eval_z_2 = x135*x40;
      dang_eval_x_3 = x42*(x44*x56 + x60*x76);
      dang_eval_y_3 = x42*(x111*x44 + x112*x60);
      dang_eval_z_3 = x41*(radial_eval*x141 + radial_eval_alpha*x140);
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

      dang_eval_x_0 = x17*(radial_eval*x79 + radial_eval_alpha*x77);
      dang_eval_y_0 = x45*(x111*x46 + x114*x60);
      dang_eval_z_0 = x142*(radial_eval_alpha*x46 + x143);
      dang_eval_x_1 = x48*x81;
      dang_eval_y_1 = x116*x48;
      dang_eval_z_1 = x47*(radial_eval*x49 + radial_eval_alpha*x144);
      dang_eval_x_2 = x0*(radial_eval*x84 + radial_eval_alpha*x82);
      dang_eval_y_2 = x50*(x111*x51 + x117*x52);
      dang_eval_z_2 = x118*x51*x56;
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
