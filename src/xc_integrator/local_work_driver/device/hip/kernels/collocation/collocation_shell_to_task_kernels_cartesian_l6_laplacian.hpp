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


__global__ __launch_bounds__(128,2) void collocation_device_shell_to_task_kernel_cartesian_laplacian_6(
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
      const auto x0 = x*x*x*x*x*x; 
      const auto x1 = radial_eval*y; 
      const auto x2 = x*x*x*x*x; 
      const auto x3 = radial_eval*z; 
      const auto x4 = x*x*x*x; 
      const auto x5 = y*y; 
      const auto x6 = x4*x5; 
      const auto x7 = x1*z; 
      const auto x8 = z*z; 
      const auto x9 = x4*x8; 
      const auto x10 = x*x*x; 
      const auto x11 = y*y*y; 
      const auto x12 = x10*x11; 
      const auto x13 = x10*x5; 
      const auto x14 = x10*x8; 
      const auto x15 = z*z*z; 
      const auto x16 = x10*x15; 
      const auto x17 = x*x; 
      const auto x18 = y*y*y*y; 
      const auto x19 = x17*x18; 
      const auto x20 = x11*x17; 
      const auto x21 = x17*x5*x8; 
      const auto x22 = x15*x17; 
      const auto x23 = z*z*z*z; 
      const auto x24 = x17*x23; 
      const auto x25 = radial_eval*x; 
      const auto x26 = y*y*y*y*y; 
      const auto x27 = x*x3; 
      const auto x28 = x11*x8; 
      const auto x29 = x15*x5; 
      const auto x30 = x*x1; 
      const auto x31 = z*z*z*z*z; 
      const auto x32 = y*y*y*y*y*y; 
      const auto x33 = x18*x8; 
      const auto x34 = x11*x15; 
      const auto x35 = x23*x5; 
      const auto x36 = z*z*z*z*z*z; 
      const auto x37 = x*x*x*x*x*x*x; 
      const auto x38 = 6.0*radial_eval; 
      const auto x39 = 5.0*radial_eval; 
      const auto x40 = radial_eval_alpha*x0 + x39*x4; 
      const auto x41 = x2*x5; 
      const auto x42 = radial_eval_alpha*x41; 
      const auto x43 = 4.0*radial_eval; 
      const auto x44 = y*z; 
      const auto x45 = radial_eval_alpha*x2; 
      const auto x46 = x10*x43 + x45; 
      const auto x47 = x2*x8; 
      const auto x48 = radial_eval_alpha*x47; 
      const auto x49 = x11*x4; 
      const auto x50 = radial_eval_alpha*x49; 
      const auto x51 = 3.0*radial_eval; 
      const auto x52 = radial_eval_alpha*x6; 
      const auto x53 = x17*x5; 
      const auto x54 = x51*x53; 
      const auto x55 = x52 + x54; 
      const auto x56 = radial_eval_alpha*x9; 
      const auto x57 = x17*x8; 
      const auto x58 = x51*x57; 
      const auto x59 = x56 + x58; 
      const auto x60 = x15*x4; 
      const auto x61 = radial_eval_alpha*x60; 
      const auto x62 = 2.0*x25; 
      const auto x63 = x10*x18; 
      const auto x64 = radial_eval_alpha*x63; 
      const auto x65 = radial_eval_alpha*x12; 
      const auto x66 = x11*x62 + x65; 
      const auto x67 = x5*x8; 
      const auto x68 = radial_eval_alpha*x10*x5*x8; 
      const auto x69 = radial_eval_alpha*x16; 
      const auto x70 = x15*x62 + x69; 
      const auto x71 = x10*x23; 
      const auto x72 = radial_eval_alpha*x71; 
      const auto x73 = radial_eval*x26; 
      const auto x74 = x17*x26; 
      const auto x75 = radial_eval_alpha*x74; 
      const auto x76 = radial_eval*x18; 
      const auto x77 = radial_eval_alpha*x19; 
      const auto x78 = x76 + x77; 
      const auto x79 = radial_eval_alpha*x11*x17*x8; 
      const auto x80 = radial_eval_alpha*x15*x17*x5; 
      const auto x81 = radial_eval*x23; 
      const auto x82 = radial_eval_alpha*x24; 
      const auto x83 = x81 + x82; 
      const auto x84 = radial_eval*x31; 
      const auto x85 = x17*x31; 
      const auto x86 = radial_eval_alpha*x85; 
      const auto x87 = radial_eval_alpha*x; 
      const auto x88 = x26*x87*z; 
      const auto x89 = x31*x87*y; 
      const auto x90 = radial_eval_alpha*y; 
      const auto x91 = radial_eval*x2; 
      const auto x92 = radial_eval_alpha*x2*x44; 
      const auto x93 = 2.0*x1; 
      const auto x94 = radial_eval*x4; 
      const auto x95 = x52 + x94; 
      const auto x96 = x10*x93 + x65; 
      const auto x97 = x54 + x77; 
      const auto x98 = radial_eval_alpha*x32 + x18*x39; 
      const auto x99 = x*z; 
      const auto x100 = radial_eval_alpha*x26; 
      const auto x101 = x100 + x11*x43; 
      const auto x102 = radial_eval_alpha*x33; 
      const auto x103 = x51*x67; 
      const auto x104 = x102 + x103; 
      const auto x105 = radial_eval_alpha*x34; 
      const auto x106 = x105 + x15*x93; 
      const auto x107 = radial_eval_alpha*x35; 
      const auto x108 = y*y*y*y*y*y*y; 
      const auto x109 = x26*x8; 
      const auto x110 = radial_eval_alpha*x109; 
      const auto x111 = x15*x18; 
      const auto x112 = radial_eval_alpha*x111; 
      const auto x113 = x11*x23; 
      const auto x114 = radial_eval_alpha*x113; 
      const auto x115 = x31*x5; 
      const auto x116 = radial_eval_alpha*x115; 
      const auto x117 = radial_eval_alpha*z; 
      const auto x118 = 2.0*x3; 
      const auto x119 = x10*x118 + x69; 
      const auto x120 = x58 + x82; 
      const auto x121 = x105 + x11*x118; 
      const auto x122 = x103 + x107; 
      const auto x123 = x*y; 
      const auto x124 = radial_eval_alpha*x31; 
      const auto x125 = x124 + x15*x43; 
      const auto x126 = radial_eval_alpha*x36 + x23*x39; 
      const auto x127 = z*z*z*z*z*z*z; 
      const auto x128 = 30.0*radial_eval; 
      const auto x129 = 12.0*radial_eval_alpha; 
      const auto x130 = radial_eval_alpha + radial_eval_alpha_squared*x17; 
      const auto x131 = x0*x129 + x0*x130 + x128*x4; 
      const auto x132 = 10.0*radial_eval_alpha; 
      const auto x133 = 20.0*radial_eval; 
      const auto x134 = x10*x133 + x130*x2; 
      const auto x135 = x132*x2 + x134; 
      const auto x136 = 8.0*radial_eval_alpha; 
      const auto x137 = 12.0*radial_eval; 
      const auto x138 = x137*x53; 
      const auto x139 = x130*x4*x5 + x138; 
      const auto x140 = x130*x4 + x137*x17; 
      const auto x141 = x137*x57; 
      const auto x142 = x130*x4*x8 + x141; 
      const auto x143 = 6.0*radial_eval_alpha; 
      const auto x144 = x12*x143; 
      const auto x145 = 6.0*x25; 
      const auto x146 = x11*(x10*x130 + x145); 
      const auto x147 = x145*x5; 
      const auto x148 = x10*x130*x5 + x147; 
      const auto x149 = x145*x8; 
      const auto x150 = x10*x130*x8 + x149; 
      const auto x151 = x143*x16; 
      const auto x152 = x15*(x10*x130 + x145); 
      const auto x153 = 4.0*radial_eval_alpha; 
      const auto x154 = 2.0*radial_eval; 
      const auto x155 = x154*x18; 
      const auto x156 = x130*x17*x18 + x155; 
      const auto x157 = x153*x20; 
      const auto x158 = x11*x154; 
      const auto x159 = x11*x130*x17 + x158; 
      const auto x160 = x153*x21; 
      const auto x161 = x130*x17*x5*x8 + x154*x67; 
      const auto x162 = x153*x22; 
      const auto x163 = x15*x154; 
      const auto x164 = x130*x15*x17 + x163; 
      const auto x165 = x154*x23; 
      const auto x166 = x130*x17*x23 + x165; 
      const auto x167 = 2.0*radial_eval_alpha; 
      const auto x168 = x167*x26; 
      const auto x169 = x130*x26; 
      const auto x170 = x167*x18; 
      const auto x171 = x130*x18; 
      const auto x172 = x11*x130*x8; 
      const auto x173 = x130*x15*x5; 
      const auto x174 = x167*x23; 
      const auto x175 = x130*x23; 
      const auto x176 = x167*x31; 
      const auto x177 = x130*x31; 
      const auto x178 = x130*x32; 
      const auto x179 = x130*x18*x8; 
      const auto x180 = x11*x130*x15; 
      const auto x181 = x130*x23*x5; 
      const auto x182 = x130*x36; 
      const auto x183 = radial_eval_alpha_squared*x37 + x143*x2; 
      const auto x184 = 5.0*radial_eval_alpha; 
      const auto x185 = x44*(radial_eval_alpha_squared*x0 + x184*x4); 
      const auto x186 = 8.0*x1; 
      const auto x187 = 2.0*x90; 
      const auto x188 = x12*x153; 
      const auto x189 = radial_eval_alpha_squared*x41; 
      const auto x190 = x13*x153; 
      const auto x191 = x189 + x190; 
      const auto x192 = radial_eval_alpha_squared*x47; 
      const auto x193 = x14*x153; 
      const auto x194 = x192 + x193; 
      const auto x195 = 9.0*radial_eval; 
      const auto x196 = 3.0*radial_eval_alpha; 
      const auto x197 = x187*x4; 
      const auto x198 = 6.0*x1; 
      const auto x199 = x17*x198; 
      const auto x200 = radial_eval_alpha_squared*x49; 
      const auto x201 = x196*x20 + x200; 
      const auto x202 = radial_eval_alpha_squared*x4*x5*x8; 
      const auto x203 = x196*x21; 
      const auto x204 = x202 + x203; 
      const auto x205 = radial_eval_alpha_squared*x60; 
      const auto x206 = x196*x22 + x205; 
      const auto x207 = 8.0*x25; 
      const auto x208 = 2.0*x87; 
      const auto x209 = x13*x196; 
      const auto x210 = radial_eval_alpha_squared*x63; 
      const auto x211 = x18*x208 + x210; 
      const auto x212 = x14*x187; 
      const auto x213 = radial_eval_alpha_squared*x10*x11*x8; 
      const auto x214 = x208*x28 + x213; 
      const auto x215 = radial_eval_alpha_squared*x10*x15*x5; 
      const auto x216 = x208*x29 + x215; 
      const auto x217 = radial_eval_alpha_squared*x71; 
      const auto x218 = x208*x23 + x217; 
      const auto x219 = radial_eval_alpha_squared*x74; 
      const auto x220 = x157 + x219; 
      const auto x221 = radial_eval_alpha_squared*x17*x18*x8; 
      const auto x222 = x203 + x221; 
      const auto x223 = radial_eval_alpha_squared*x11*x15*x17; 
      const auto x224 = x187*x22 + x223; 
      const auto x225 = radial_eval_alpha_squared*x17*x23*x5; 
      const auto x226 = radial_eval_alpha_squared*x85; 
      const auto x227 = radial_eval_alpha_squared*x108 + x143*x26; 
      const auto x228 = x99*(radial_eval_alpha_squared*x32 + x18*x184); 
      const auto x229 = radial_eval_alpha_squared*x109; 
      const auto x230 = x153*x28; 
      const auto x231 = x229 + x230; 
      const auto x232 = radial_eval_alpha_squared*x111; 
      const auto x233 = x196*x29 + x232; 
      const auto x234 = radial_eval_alpha_squared*x113; 
      const auto x235 = x187*x23 + x234; 
      const auto x236 = radial_eval_alpha_squared*x115; 
      const auto x237 = 8.0*x3; 
      const auto x238 = 2.0*x117; 
      const auto x239 = x153*x16; 
      const auto x240 = x238*x4; 
      const auto x241 = 6.0*x3; 
      const auto x242 = x17*x241; 
      const auto x243 = x13*x238; 
      const auto x244 = x14*x196; 
      const auto x245 = x20*x238; 
      const auto x246 = x203 + x225; 
      const auto x247 = x162 + x226; 
      const auto x248 = x18*x238; 
      const auto x249 = x196*x28; 
      const auto x250 = x153*x29; 
      const auto x251 = x236 + x250; 
      const auto x252 = x123*(radial_eval_alpha_squared*x36 + x184*x23); 
      const auto x253 = radial_eval_alpha_squared*x127 + x143*x31; 
      const auto x254 = radial_eval_alpha + radial_eval_alpha_squared*x5; 
      const auto x255 = x0*x254; 
      const auto x256 = x167*x2; 
      const auto x257 = x2*x254; 
      const auto x258 = x154*x4; 
      const auto x259 = x254*x4*x5 + x258; 
      const auto x260 = x167*x4; 
      const auto x261 = x254*x4; 
      const auto x262 = x254*x4*x8; 
      const auto x263 = x10*(x11*x254 + x198); 
      const auto x264 = x10*x154; 
      const auto x265 = x10*x254*x5 + x264; 
      const auto x266 = x10*x254*x8; 
      const auto x267 = x10*x15*x254; 
      const auto x268 = x138 + x17*x18*x254; 
      const auto x269 = x11*x17*x254 + x199; 
      const auto x270 = x154*x57 + x17*x254*x5*x8; 
      const auto x271 = x15*x17*x254; 
      const auto x272 = x17*x23*x254; 
      const auto x273 = x11*x133 + x254*x26; 
      const auto x274 = x132*x26 + x273; 
      const auto x275 = x137*x5 + x18*x254; 
      const auto x276 = x198*x8; 
      const auto x277 = x11*x254*x8 + x276; 
      const auto x278 = x15*x254*x5 + x163; 
      const auto x279 = x23*x254; 
      const auto x280 = x254*x31; 
      const auto x281 = x128*x18 + x129*x32 + x254*x32; 
      const auto x282 = x137*x67; 
      const auto x283 = x18*x254*x8 + x282; 
      const auto x284 = x143*x34; 
      const auto x285 = x15*(x11*x254 + x198); 
      const auto x286 = x165 + x23*x254*x5; 
      const auto x287 = x254*x36; 
      const auto x288 = x241*x5; 
      const auto x289 = x153*x34; 
      const auto x290 = radial_eval_alpha + radial_eval_alpha_squared*x8; 
      const auto x291 = x0*x290; 
      const auto x292 = x2*x290; 
      const auto x293 = x290*x4*x5; 
      const auto x294 = x290*x4; 
      const auto x295 = x258 + x290*x4*x8; 
      const auto x296 = x10*x11*x290; 
      const auto x297 = x10*x290*x5; 
      const auto x298 = x10*x290*x8 + x264; 
      const auto x299 = x10*(x15*x290 + x241); 
      const auto x300 = x17*x18*x290; 
      const auto x301 = x11*x17*x290; 
      const auto x302 = x154*x53 + x17*x290*x5*x8; 
      const auto x303 = x15*x17*x290 + x242; 
      const auto x304 = x141 + x17*x23*x290; 
      const auto x305 = x26*x290; 
      const auto x306 = x18*x290; 
      const auto x307 = x11*x290*x8 + x158; 
      const auto x308 = x15*x290*x5 + x288; 
      const auto x309 = x137*x8 + x23*x290; 
      const auto x310 = x133*x15 + x290*x31; 
      const auto x311 = x132*x31 + x310; 
      const auto x312 = x290*x32; 
      const auto x313 = x155 + x18*x290*x8; 
      const auto x314 = x11*(x15*x290 + x241); 
      const auto x315 = x23*x290*x5 + x282; 
      const auto x316 = x128*x23 + x129*x36 + x290*x36; 
      const auto x317 = x129*x2 + x134 + x257 + x292; 
      const auto x318 = x129*x26 + x169 + x273 + x305; 
      const auto x319 = x129*x31 + x177 + x280 + x310; 


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
      basis_eval[ipt + 10*npts] = radial_eval*x19;
      basis_eval[ipt + 11*npts] = x20*x3;
      basis_eval[ipt + 12*npts] = radial_eval*x21;
      basis_eval[ipt + 13*npts] = x1*x22;
      basis_eval[ipt + 14*npts] = radial_eval*x24;
      basis_eval[ipt + 15*npts] = x25*x26;
      basis_eval[ipt + 16*npts] = x18*x27;
      basis_eval[ipt + 17*npts] = x25*x28;
      basis_eval[ipt + 18*npts] = x25*x29;
      basis_eval[ipt + 19*npts] = x23*x30;
      basis_eval[ipt + 20*npts] = x25*x31;
      basis_eval[ipt + 21*npts] = radial_eval*x32;
      basis_eval[ipt + 22*npts] = x26*x3;
      basis_eval[ipt + 23*npts] = radial_eval*x33;
      basis_eval[ipt + 24*npts] = radial_eval*x34;
      basis_eval[ipt + 25*npts] = radial_eval*x35;
      basis_eval[ipt + 26*npts] = x1*x31;
      basis_eval[ipt + 27*npts] = radial_eval*x36;


    
      // Evaluate first derivative of bfn wrt x
      basis_x_eval[ipt + 0*npts] = radial_eval_alpha*x37 + x2*x38;
      basis_x_eval[ipt + 1*npts] = x40*y;
      basis_x_eval[ipt + 2*npts] = x40*z;
      basis_x_eval[ipt + 3*npts] = x13*x43 + x42;
      basis_x_eval[ipt + 4*npts] = x44*x46;
      basis_x_eval[ipt + 5*npts] = x14*x43 + x48;
      basis_x_eval[ipt + 6*npts] = x20*x51 + x50;
      basis_x_eval[ipt + 7*npts] = x55*z;
      basis_x_eval[ipt + 8*npts] = x59*y;
      basis_x_eval[ipt + 9*npts] = x22*x51 + x61;
      basis_x_eval[ipt + 10*npts] = x18*x62 + x64;
      basis_x_eval[ipt + 11*npts] = x66*z;
      basis_x_eval[ipt + 12*npts] = x62*x67 + x68;
      basis_x_eval[ipt + 13*npts] = x70*y;
      basis_x_eval[ipt + 14*npts] = x23*x62 + x72;
      basis_x_eval[ipt + 15*npts] = x73 + x75;
      basis_x_eval[ipt + 16*npts] = x78*z;
      basis_x_eval[ipt + 17*npts] = radial_eval*x28 + x79;
      basis_x_eval[ipt + 18*npts] = radial_eval*x29 + x80;
      basis_x_eval[ipt + 19*npts] = x83*y;
      basis_x_eval[ipt + 20*npts] = x84 + x86;
      basis_x_eval[ipt + 21*npts] = x32*x87;
      basis_x_eval[ipt + 22*npts] = x88;
      basis_x_eval[ipt + 23*npts] = x33*x87;
      basis_x_eval[ipt + 24*npts] = x34*x87;
      basis_x_eval[ipt + 25*npts] = x35*x87;
      basis_x_eval[ipt + 26*npts] = x89;
      basis_x_eval[ipt + 27*npts] = x36*x87;

      // Evaluate first derivative of bfn wrt y
      basis_y_eval[ipt + 0*npts] = x0*x90;
      basis_y_eval[ipt + 1*npts] = x42 + x91;
      basis_y_eval[ipt + 2*npts] = x92;
      basis_y_eval[ipt + 3*npts] = x4*x93 + x50;
      basis_y_eval[ipt + 4*npts] = x95*z;
      basis_y_eval[ipt + 5*npts] = x9*x90;
      basis_y_eval[ipt + 6*npts] = x13*x51 + x64;
      basis_y_eval[ipt + 7*npts] = x96*z;
      basis_y_eval[ipt + 8*npts] = radial_eval*x14 + x68;
      basis_y_eval[ipt + 9*npts] = x16*x90;
      basis_y_eval[ipt + 10*npts] = x20*x43 + x75;
      basis_y_eval[ipt + 11*npts] = x97*z;
      basis_y_eval[ipt + 12*npts] = x57*x93 + x79;
      basis_y_eval[ipt + 13*npts] = radial_eval*x22 + x80;
      basis_y_eval[ipt + 14*npts] = x24*x90;
      basis_y_eval[ipt + 15*npts] = x*x98;
      basis_y_eval[ipt + 16*npts] = x101*x99;
      basis_y_eval[ipt + 17*npts] = x*x104;
      basis_y_eval[ipt + 18*npts] = x*x106;
      basis_y_eval[ipt + 19*npts] = x*(x107 + x81);
      basis_y_eval[ipt + 20*npts] = x89;
      basis_y_eval[ipt + 21*npts] = radial_eval_alpha*x108 + x26*x38;
      basis_y_eval[ipt + 22*npts] = x98*z;
      basis_y_eval[ipt + 23*npts] = x110 + x28*x43;
      basis_y_eval[ipt + 24*npts] = x112 + x29*x51;
      basis_y_eval[ipt + 25*npts] = x114 + x23*x93;
      basis_y_eval[ipt + 26*npts] = x116 + x84;
      basis_y_eval[ipt + 27*npts] = x36*x90;

      // Evaluate first derivative of bfn wrt z
      basis_z_eval[ipt + 0*npts] = x0*x117;
      basis_z_eval[ipt + 1*npts] = x92;
      basis_z_eval[ipt + 2*npts] = x48 + x91;
      basis_z_eval[ipt + 3*npts] = x117*x6;
      basis_z_eval[ipt + 4*npts] = y*(x56 + x94);
      basis_z_eval[ipt + 5*npts] = x118*x4 + x61;
      basis_z_eval[ipt + 6*npts] = x117*x12;
      basis_z_eval[ipt + 7*npts] = radial_eval*x13 + x68;
      basis_z_eval[ipt + 8*npts] = x119*y;
      basis_z_eval[ipt + 9*npts] = x14*x51 + x72;
      basis_z_eval[ipt + 10*npts] = x117*x19;
      basis_z_eval[ipt + 11*npts] = radial_eval*x20 + x79;
      basis_z_eval[ipt + 12*npts] = x118*x53 + x80;
      basis_z_eval[ipt + 13*npts] = x120*y;
      basis_z_eval[ipt + 14*npts] = x22*x43 + x86;
      basis_z_eval[ipt + 15*npts] = x88;
      basis_z_eval[ipt + 16*npts] = x*(x102 + x76);
      basis_z_eval[ipt + 17*npts] = x*x121;
      basis_z_eval[ipt + 18*npts] = x*x122;
      basis_z_eval[ipt + 19*npts] = x123*x125;
      basis_z_eval[ipt + 20*npts] = x*x126;
      basis_z_eval[ipt + 21*npts] = x117*x32;
      basis_z_eval[ipt + 22*npts] = x110 + x73;
      basis_z_eval[ipt + 23*npts] = x112 + x118*x18;
      basis_z_eval[ipt + 24*npts] = x114 + x28*x51;
      basis_z_eval[ipt + 25*npts] = x116 + x29*x43;
      basis_z_eval[ipt + 26*npts] = x126*y;
      basis_z_eval[ipt + 27*npts] = radial_eval_alpha*x127 + x31*x38;


      // Evaluate Laplacian of bfn 
      basis_lapl_eval[ipt + 0*npts] = x131 + x255 + x291;
      basis_lapl_eval[ipt + 1*npts] = x317*y;
      basis_lapl_eval[ipt + 2*npts] = x317*z;
      basis_lapl_eval[ipt + 3*npts] = x129*x6 + x139 + x259 + x293;
      basis_lapl_eval[ipt + 4*npts] = x44*(x129*x4 + x140 + x261 + x294);
      basis_lapl_eval[ipt + 5*npts] = x129*x9 + x142 + x262 + x295;
      basis_lapl_eval[ipt + 6*npts] = x12*x129 + x146 + x263 + x296;
      basis_lapl_eval[ipt + 7*npts] = z*(x129*x13 + x148 + x265 + x297);
      basis_lapl_eval[ipt + 8*npts] = y*(x129*x14 + x150 + x266 + x298);
      basis_lapl_eval[ipt + 9*npts] = x129*x16 + x152 + x267 + x299;
      basis_lapl_eval[ipt + 10*npts] = x129*x19 + x156 + x268 + x300;
      basis_lapl_eval[ipt + 11*npts] = z*(x129*x20 + x159 + x269 + x301);
      basis_lapl_eval[ipt + 12*npts] = x129*x21 + x161 + x270 + x302;
      basis_lapl_eval[ipt + 13*npts] = y*(x129*x22 + x164 + x271 + x303);
      basis_lapl_eval[ipt + 14*npts] = x129*x24 + x166 + x272 + x304;
      basis_lapl_eval[ipt + 15*npts] = x*x318;
      basis_lapl_eval[ipt + 16*npts] = x99*(x129*x18 + x171 + x275 + x306);
      basis_lapl_eval[ipt + 17*npts] = x*(x129*x28 + x172 + x277 + x307);
      basis_lapl_eval[ipt + 18*npts] = x*(x129*x29 + x173 + x278 + x308);
      basis_lapl_eval[ipt + 19*npts] = x123*(x129*x23 + x175 + x279 + x309);
      basis_lapl_eval[ipt + 20*npts] = x*x319;
      basis_lapl_eval[ipt + 21*npts] = x178 + x281 + x312;
      basis_lapl_eval[ipt + 22*npts] = x318*z;
      basis_lapl_eval[ipt + 23*npts] = x129*x33 + x179 + x283 + x313;
      basis_lapl_eval[ipt + 24*npts] = x129*x34 + x180 + x285 + x314;
      basis_lapl_eval[ipt + 25*npts] = x129*x35 + x181 + x286 + x315;
      basis_lapl_eval[ipt + 26*npts] = x319*y;
      basis_lapl_eval[ipt + 27*npts] = x182 + x287 + x316;





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
      ang_eval_2 = radial_eval*x19;
      ang_eval_3 = x20*x3;
      basis_eval[ipt + 8*npts] = ang_eval_0;
      basis_eval[ipt + 9*npts] = ang_eval_1;
      basis_eval[ipt + 10*npts] = ang_eval_2;
      basis_eval[ipt + 11*npts] = ang_eval_3;

      ang_eval_0 = radial_eval*x21;
      ang_eval_1 = x1*x22;
      ang_eval_2 = radial_eval*x24;
      ang_eval_3 = x25*x26;
      basis_eval[ipt + 12*npts] = ang_eval_0;
      basis_eval[ipt + 13*npts] = ang_eval_1;
      basis_eval[ipt + 14*npts] = ang_eval_2;
      basis_eval[ipt + 15*npts] = ang_eval_3;

      ang_eval_0 = x18*x27;
      ang_eval_1 = x25*x28;
      ang_eval_2 = x25*x29;
      ang_eval_3 = x23*x30;
      basis_eval[ipt + 16*npts] = ang_eval_0;
      basis_eval[ipt + 17*npts] = ang_eval_1;
      basis_eval[ipt + 18*npts] = ang_eval_2;
      basis_eval[ipt + 19*npts] = ang_eval_3;

      ang_eval_0 = x25*x31;
      ang_eval_1 = radial_eval*x32;
      ang_eval_2 = x26*x3;
      ang_eval_3 = radial_eval*x33;
      basis_eval[ipt + 20*npts] = ang_eval_0;
      basis_eval[ipt + 21*npts] = ang_eval_1;
      basis_eval[ipt + 22*npts] = ang_eval_2;
      basis_eval[ipt + 23*npts] = ang_eval_3;

      ang_eval_0 = radial_eval*x34;
      ang_eval_1 = radial_eval*x35;
      ang_eval_2 = x1*x31;
      ang_eval_3 = radial_eval*x36;
      basis_eval[ipt + 24*npts] = ang_eval_0;
      basis_eval[ipt + 25*npts] = ang_eval_1;
      basis_eval[ipt + 26*npts] = ang_eval_2;
      basis_eval[ipt + 27*npts] = ang_eval_3;


      double dang_eval_x_0, dang_eval_y_0, dang_eval_z_0;
      double dang_eval_x_1, dang_eval_y_1, dang_eval_z_1;
      double dang_eval_x_2, dang_eval_y_2, dang_eval_z_2;
      double dang_eval_x_3, dang_eval_y_3, dang_eval_z_3;

      dang_eval_x_0 = radial_eval_alpha*x37 + x2*x38;
      dang_eval_y_0 = x0*x90;
      dang_eval_z_0 = x0*x117;
      dang_eval_x_1 = x40*y;
      dang_eval_y_1 = x42 + x91;
      dang_eval_z_1 = x92;
      dang_eval_x_2 = x40*z;
      dang_eval_y_2 = x92;
      dang_eval_z_2 = x48 + x91;
      dang_eval_x_3 = x13*x43 + x42;
      dang_eval_y_3 = x4*x93 + x50;
      dang_eval_z_3 = x117*x6;
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

      dang_eval_x_0 = x44*x46;
      dang_eval_y_0 = x95*z;
      dang_eval_z_0 = y*(x56 + x94);
      dang_eval_x_1 = x14*x43 + x48;
      dang_eval_y_1 = x9*x90;
      dang_eval_z_1 = x118*x4 + x61;
      dang_eval_x_2 = x20*x51 + x50;
      dang_eval_y_2 = x13*x51 + x64;
      dang_eval_z_2 = x117*x12;
      dang_eval_x_3 = x55*z;
      dang_eval_y_3 = x96*z;
      dang_eval_z_3 = radial_eval*x13 + x68;
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

      dang_eval_x_0 = x59*y;
      dang_eval_y_0 = radial_eval*x14 + x68;
      dang_eval_z_0 = x119*y;
      dang_eval_x_1 = x22*x51 + x61;
      dang_eval_y_1 = x16*x90;
      dang_eval_z_1 = x14*x51 + x72;
      dang_eval_x_2 = x18*x62 + x64;
      dang_eval_y_2 = x20*x43 + x75;
      dang_eval_z_2 = x117*x19;
      dang_eval_x_3 = x66*z;
      dang_eval_y_3 = x97*z;
      dang_eval_z_3 = radial_eval*x20 + x79;
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

      dang_eval_x_0 = x62*x67 + x68;
      dang_eval_y_0 = x57*x93 + x79;
      dang_eval_z_0 = x118*x53 + x80;
      dang_eval_x_1 = x70*y;
      dang_eval_y_1 = radial_eval*x22 + x80;
      dang_eval_z_1 = x120*y;
      dang_eval_x_2 = x23*x62 + x72;
      dang_eval_y_2 = x24*x90;
      dang_eval_z_2 = x22*x43 + x86;
      dang_eval_x_3 = x73 + x75;
      dang_eval_y_3 = x*x98;
      dang_eval_z_3 = x88;
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

      dang_eval_x_0 = x78*z;
      dang_eval_y_0 = x101*x99;
      dang_eval_z_0 = x*(x102 + x76);
      dang_eval_x_1 = radial_eval*x28 + x79;
      dang_eval_y_1 = x*x104;
      dang_eval_z_1 = x*x121;
      dang_eval_x_2 = radial_eval*x29 + x80;
      dang_eval_y_2 = x*x106;
      dang_eval_z_2 = x*x122;
      dang_eval_x_3 = x83*y;
      dang_eval_y_3 = x*(x107 + x81);
      dang_eval_z_3 = x123*x125;
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

      dang_eval_x_0 = x84 + x86;
      dang_eval_y_0 = x89;
      dang_eval_z_0 = x*x126;
      dang_eval_x_1 = x32*x87;
      dang_eval_y_1 = radial_eval_alpha*x108 + x26*x38;
      dang_eval_z_1 = x117*x32;
      dang_eval_x_2 = x88;
      dang_eval_y_2 = x98*z;
      dang_eval_z_2 = x110 + x73;
      dang_eval_x_3 = x33*x87;
      dang_eval_y_3 = x110 + x28*x43;
      dang_eval_z_3 = x112 + x118*x18;
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

      dang_eval_x_0 = x34*x87;
      dang_eval_y_0 = x112 + x29*x51;
      dang_eval_z_0 = x114 + x28*x51;
      dang_eval_x_1 = x35*x87;
      dang_eval_y_1 = x114 + x23*x93;
      dang_eval_z_1 = x116 + x29*x43;
      dang_eval_x_2 = x89;
      dang_eval_y_2 = x116 + x84;
      dang_eval_z_2 = x126*y;
      dang_eval_x_3 = x36*x87;
      dang_eval_y_3 = x36*x90;
      dang_eval_z_3 = radial_eval_alpha*x127 + x31*x38;
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
