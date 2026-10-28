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


__global__ __launch_bounds__(128,2) void collocation_device_shell_to_task_kernel_cartesian_lapgrad_5(
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
    auto* __restrict__ basis_xx_eval = task->d2bfxx + shoff;
    auto* __restrict__ basis_xy_eval = task->d2bfxy + shoff;
    auto* __restrict__ basis_xz_eval = task->d2bfxz + shoff;
    auto* __restrict__ basis_yy_eval = task->d2bfyy + shoff;
    auto* __restrict__ basis_yz_eval = task->d2bfyz + shoff;
    auto* __restrict__ basis_zz_eval = task->d2bfzz + shoff;
    auto* __restrict__ basis_lapl_eval = task->d2bflapl + shoff;
    auto* __restrict__ basis_lapl_x_eval = task->d3bflapl_x + shoff;
    auto* __restrict__ basis_lapl_y_eval = task->d3bflapl_y + shoff;
    auto* __restrict__ basis_lapl_z_eval = task->d3bflapl_z + shoff;

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
      double radial_eval_alpha_squared = 0.;
      double radial_eval_alpha_cubed = 0.;

      #pragma unroll 1
      for( uint32_t i = 0; i < nprim; ++i ) {
        const auto a = my_alpha[i];
        const auto e = my_coeff[i] * std::exp( - a * rsq );

        radial_eval += e;
        radial_eval_alpha += a * e;
        radial_eval_alpha_squared += a * a * e;
        radial_eval_alpha_cubed += a * a * a * e;
      }

      radial_eval_alpha *= -2;
      radial_eval_alpha_squared *= 4;
      radial_eval_alpha_cubed *= -8;

      // Common Subexpressions
      const auto x0 = x*x*x*x*x; 
      const auto x1 = radial_eval*y; 
      const auto x2 = x*x*x*x; 
      const auto x3 = radial_eval*z; 
      const auto x4 = x*x*x; 
      const auto x5 = y*y; 
      const auto x6 = x5; 
      const auto x7 = x4*x6; 
      const auto x8 = x1*z; 
      const auto x9 = z*z; 
      const auto x10 = x9; 
      const auto x11 = x10*x4; 
      const auto x12 = x*x; 
      const auto x13 = x12; 
      const auto x14 = y*y*y; 
      const auto x15 = x13*x14; 
      const auto x16 = x13*x6; 
      const auto x17 = x10*x13; 
      const auto x18 = z*z*z; 
      const auto x19 = x13*x18; 
      const auto x20 = radial_eval*x; 
      const auto x21 = y*y*y*y; 
      const auto x22 = x*x3; 
      const auto x23 = x10*x6; 
      const auto x24 = x*x1; 
      const auto x25 = z*z*z*z; 
      const auto x26 = y*y*y*y*y; 
      const auto x27 = x10*x14; 
      const auto x28 = x18*x6; 
      const auto x29 = z*z*z*z*z; 
      const auto x30 = x*x*x*x*x*x; 
      const auto x31 = 5.0*radial_eval; 
      const auto x32 = 4.0*radial_eval; 
      const auto x33 = radial_eval_alpha*x0 + x32*x4; 
      const auto x34 = x2*x6; 
      const auto x35 = radial_eval_alpha*x34; 
      const auto x36 = 3.0*radial_eval; 
      const auto x37 = x16*x36; 
      const auto x38 = y*z; 
      const auto x39 = radial_eval_alpha*x2; 
      const auto x40 = x13*x36 + x39; 
      const auto x41 = x10*x2; 
      const auto x42 = radial_eval_alpha*x41; 
      const auto x43 = x17*x36; 
      const auto x44 = 2.0*x20; 
      const auto x45 = x14*x4; 
      const auto x46 = radial_eval_alpha*x45; 
      const auto x47 = radial_eval_alpha*x7; 
      const auto x48 = x44*x6 + x47; 
      const auto x49 = radial_eval_alpha*x11; 
      const auto x50 = x10*x44 + x49; 
      const auto x51 = x18*x4; 
      const auto x52 = radial_eval_alpha*x51; 
      const auto x53 = radial_eval*x21; 
      const auto x54 = x13*x21; 
      const auto x55 = radial_eval_alpha*x54; 
      const auto x56 = radial_eval*x14; 
      const auto x57 = radial_eval_alpha*x15; 
      const auto x58 = x56 + x57; 
      const auto x59 = x10*x13*x6; 
      const auto x60 = radial_eval_alpha*x59; 
      const auto x61 = radial_eval*x18; 
      const auto x62 = radial_eval_alpha*x19; 
      const auto x63 = x61 + x62; 
      const auto x64 = radial_eval*x25; 
      const auto x65 = x13*x25; 
      const auto x66 = radial_eval_alpha*x65; 
      const auto x67 = radial_eval_alpha*x; 
      const auto x68 = x21*x67*z; 
      const auto x69 = x25*x67*y; 
      const auto x70 = radial_eval_alpha*y; 
      const auto x71 = radial_eval*x2; 
      const auto x72 = radial_eval_alpha*x2*x38; 
      const auto x73 = 2.0*x1; 
      const auto x74 = radial_eval*x4; 
      const auto x75 = x47 + x74; 
      const auto x76 = x13*x73 + x57; 
      const auto x77 = radial_eval_alpha*x26 + x14*x32; 
      const auto x78 = x*z; 
      const auto x79 = radial_eval_alpha*x21; 
      const auto x80 = x36*x6 + x79; 
      const auto x81 = radial_eval_alpha*x27; 
      const auto x82 = x10*x73 + x81; 
      const auto x83 = radial_eval_alpha*x28; 
      const auto x84 = y*y*y*y*y*y; 
      const auto x85 = x10*x21; 
      const auto x86 = radial_eval_alpha*x85; 
      const auto x87 = x23*x36; 
      const auto x88 = x14*x18; 
      const auto x89 = radial_eval_alpha*x88; 
      const auto x90 = x25*x6; 
      const auto x91 = radial_eval_alpha*x90; 
      const auto x92 = radial_eval_alpha*z; 
      const auto x93 = 2.0*x3; 
      const auto x94 = x13*x93 + x62; 
      const auto x95 = x6*x93 + x83; 
      const auto x96 = x*y; 
      const auto x97 = radial_eval_alpha*x25; 
      const auto x98 = x10*x36 + x97; 
      const auto x99 = radial_eval_alpha*x29 + x18*x32; 
      const auto x100 = z*z*z*z*z*z; 
      const auto x101 = 20.0*radial_eval; 
      const auto x102 = 10.0*radial_eval_alpha; 
      const auto x103 = radial_eval_alpha + radial_eval_alpha_squared*x13; 
      const auto x104 = x0*x102 + x0*x103 + x101*x4; 
      const auto x105 = 8.0*radial_eval_alpha; 
      const auto x106 = x103*x2; 
      const auto x107 = 12.0*radial_eval; 
      const auto x108 = x106 + x107*x13; 
      const auto x109 = x105*x2 + x108; 
      const auto x110 = 6.0*radial_eval_alpha; 
      const auto x111 = 6.0*x20; 
      const auto x112 = x111*x6; 
      const auto x113 = x103*x4*x6 + x112; 
      const auto x114 = x103*x4; 
      const auto x115 = x111 + x114; 
      const auto x116 = x10*x111; 
      const auto x117 = x10*x103*x4 + x116; 
      const auto x118 = 4.0*radial_eval_alpha; 
      const auto x119 = x118*x15; 
      const auto x120 = 2.0*radial_eval; 
      const auto x121 = x120*x14; 
      const auto x122 = x103*x13*x14 + x121; 
      const auto x123 = x118*x16; 
      const auto x124 = x120*x6; 
      const auto x125 = x103*x13*x6; 
      const auto x126 = x124 + x125; 
      const auto x127 = x118*x17; 
      const auto x128 = x10*x120; 
      const auto x129 = x10*x103*x13; 
      const auto x130 = x128 + x129; 
      const auto x131 = x118*x19; 
      const auto x132 = x120*x18; 
      const auto x133 = x103*x13*x18 + x132; 
      const auto x134 = 2.0*radial_eval_alpha; 
      const auto x135 = x134*x21; 
      const auto x136 = x103*x21; 
      const auto x137 = x134*x14; 
      const auto x138 = x103*x14; 
      const auto x139 = x134*x23; 
      const auto x140 = x10*x103*x6; 
      const auto x141 = x134*x18; 
      const auto x142 = x103*x18; 
      const auto x143 = x134*x25; 
      const auto x144 = x103*x25; 
      const auto x145 = x103*x26; 
      const auto x146 = x10*x103*x14; 
      const auto x147 = x103*x18*x6; 
      const auto x148 = x103*x29; 
      const auto x149 = 5.0*radial_eval_alpha; 
      const auto x150 = radial_eval_alpha_squared*x30 + x149*x2; 
      const auto x151 = x118*x7; 
      const auto x152 = x38*(radial_eval_alpha_squared*x0 + x118*x4); 
      const auto x153 = 6.0*x1; 
      const auto x154 = x13*x153; 
      const auto x155 = 2.0*x70; 
      const auto x156 = 3.0*radial_eval_alpha; 
      const auto x157 = radial_eval_alpha_squared*x34; 
      const auto x158 = x156*x16; 
      const auto x159 = x157 + x158; 
      const auto x160 = radial_eval_alpha_squared*x41; 
      const auto x161 = x156*x17; 
      const auto x162 = x160 + x161; 
      const auto x163 = 2.0*x67; 
      const auto x164 = x155*x4; 
      const auto x165 = x14*x163; 
      const auto x166 = radial_eval_alpha_squared*x45; 
      const auto x167 = x165 + x166; 
      const auto x168 = radial_eval_alpha_squared*x10*x4*x6; 
      const auto x169 = x163*x23 + x168; 
      const auto x170 = x163*x18; 
      const auto x171 = radial_eval_alpha_squared*x51; 
      const auto x172 = x170 + x171; 
      const auto x173 = radial_eval_alpha_squared*x54; 
      const auto x174 = x158 + x173; 
      const auto x175 = radial_eval_alpha_squared*x10*x13*x14; 
      const auto x176 = x155*x17 + x175; 
      const auto x177 = radial_eval_alpha_squared*x13*x18*x6; 
      const auto x178 = radial_eval_alpha_squared*x65; 
      const auto x179 = radial_eval_alpha_squared*x84 + x149*x21; 
      const auto x180 = x78*(radial_eval_alpha_squared*x26 + x118*x14); 
      const auto x181 = radial_eval_alpha_squared*x85; 
      const auto x182 = x156*x23; 
      const auto x183 = x181 + x182; 
      const auto x184 = x155*x18; 
      const auto x185 = radial_eval_alpha_squared*x88; 
      const auto x186 = x184 + x185; 
      const auto x187 = radial_eval_alpha_squared*x90; 
      const auto x188 = x11*x118; 
      const auto x189 = 6.0*x3; 
      const auto x190 = x13*x189; 
      const auto x191 = 2.0*x92; 
      const auto x192 = x191*x4; 
      const auto x193 = x16*x191 + x177; 
      const auto x194 = x161 + x178; 
      const auto x195 = x14*x191; 
      const auto x196 = x182 + x187; 
      const auto x197 = x96*(radial_eval_alpha_squared*x29 + x118*x18); 
      const auto x198 = radial_eval_alpha_squared*x100 + x149*x25; 
      const auto x199 = radial_eval_alpha + radial_eval_alpha_squared*x6; 
      const auto x200 = x0*x199; 
      const auto x201 = x134*x2; 
      const auto x202 = x199*x2; 
      const auto x203 = x120*x4; 
      const auto x204 = x199*x4*x6 + x203; 
      const auto x205 = x134*x4; 
      const auto x206 = x199*x4; 
      const auto x207 = x10*x199*x4; 
      const auto x208 = x13*x14*x199 + x154; 
      const auto x209 = x120*x13; 
      const auto x210 = x13*x199*x6; 
      const auto x211 = x209 + x210; 
      const auto x212 = x134*x17; 
      const auto x213 = x10*x13*x199; 
      const auto x214 = x13*x18*x199; 
      const auto x215 = x199*x21; 
      const auto x216 = x107*x6 + x215; 
      const auto x217 = x105*x21 + x216; 
      const auto x218 = x14*x199; 
      const auto x219 = x153 + x218; 
      const auto x220 = x118*x23; 
      const auto x221 = x10*x199*x6; 
      const auto x222 = x128 + x221; 
      const auto x223 = x18*x199; 
      const auto x224 = x199*x25; 
      const auto x225 = x101*x14 + x102*x26 + x199*x26; 
      const auto x226 = x10*x153; 
      const auto x227 = x10*x14*x199 + x226; 
      const auto x228 = x118*x28; 
      const auto x229 = x132 + x18*x199*x6; 
      const auto x230 = x199*x29; 
      const auto x231 = x118*x27; 
      const auto x232 = x189*x6; 
      const auto x233 = radial_eval_alpha + radial_eval_alpha_squared*x10; 
      const auto x234 = x0*x233; 
      const auto x235 = x2*x233; 
      const auto x236 = x233*x4*x6; 
      const auto x237 = x233*x4; 
      const auto x238 = x10*x233*x4 + x203; 
      const auto x239 = x13*x14*x233; 
      const auto x240 = x134*x16; 
      const auto x241 = x13*x233*x6; 
      const auto x242 = x10*x13*x233; 
      const auto x243 = x209 + x242; 
      const auto x244 = x13*x18*x233 + x190; 
      const auto x245 = x21*x233; 
      const auto x246 = x14*x233; 
      const auto x247 = x10*x233*x6; 
      const auto x248 = x124 + x247; 
      const auto x249 = x18*x233; 
      const auto x250 = x189 + x249; 
      const auto x251 = x233*x25; 
      const auto x252 = x10*x107 + x251; 
      const auto x253 = x105*x25 + x252; 
      const auto x254 = x233*x26; 
      const auto x255 = x10*x14*x233 + x121; 
      const auto x256 = x18*x233*x6 + x232; 
      const auto x257 = x101*x18 + x102*x29 + x233*x29; 
      const auto x258 = x102*x2 + x108 + x202 + x235; 
      const auto x259 = x115 + x237; 
      const auto x260 = x126 + x211; 
      const auto x261 = x130 + x243; 
      const auto x262 = x102*x21 + x136 + x216 + x245; 
      const auto x263 = x219 + x246; 
      const auto x264 = x222 + x248; 
      const auto x265 = x223 + x250; 
      const auto x266 = x102*x25 + x144 + x224 + x252; 
      const auto x267 = 60.0*radial_eval; 
      const auto x268 = 60.0*radial_eval_alpha; 
      const auto x269 = radial_eval_alpha_cubed*x6 + radial_eval_alpha_squared; 
      const auto x270 = x0*x269; 
      const auto x271 = radial_eval_alpha_cubed*x10 + radial_eval_alpha_squared; 
      const auto x272 = x0*x271; 
      const auto x273 = radial_eval_alpha_squared*x; 
      const auto x274 = radial_eval_alpha_cubed*x4 + 3.0*x273; 
      const auto x275 = 44.0*radial_eval_alpha; 
      const auto x276 = 2.0*radial_eval_alpha_squared; 
      const auto x277 = x2*x269; 
      const auto x278 = x2*x271; 
      const auto x279 = x*x277 + x*x278 + x0*x276 + 12.0*x114 + x2*x274 + 24.0*x20 + 4.0*x206 + 4.0*x237 + x275*x4; 
      const auto x280 = 4.0*radial_eval_alpha_squared; 
      const auto x281 = x269*x4*x6; 
      const auto x282 = x271*x4*x6; 
      const auto x283 = 6.0*radial_eval; 
      const auto x284 = x13*x283; 
      const auto x285 = x201 + x284; 
      const auto x286 = x283*x6; 
      const auto x287 = 3.0*x241; 
      const auto x288 = 30.0*radial_eval_alpha; 
      const auto x289 = x16*x288 + x286 + x287; 
      const auto x290 = x103*x13; 
      const auto x291 = x13*x199; 
      const auto x292 = x13*x233; 
      const auto x293 = x269*x4; 
      const auto x294 = x271*x4; 
      const auto x295 = x10*x269*x4; 
      const auto x296 = x10*x271*x4; 
      const auto x297 = x10*x283; 
      const auto x298 = 3.0*x213; 
      const auto x299 = x17*x288 + x297 + x298; 
      const auto x300 = 2.0*x; 
      const auto x301 = 6.0*x70; 
      const auto x302 = 6.0*x; 
      const auto x303 = 18.0*x67; 
      const auto x304 = x13*x14*x269; 
      const auto x305 = x13*x14*x271; 
      const auto x306 = 6.0*radial_eval_alpha_squared; 
      const auto x307 = 12.0*x24 + x306*x45; 
      const auto x308 = x199*x6; 
      const auto x309 = x233*x6; 
      const auto x310 = x103*x6; 
      const auto x311 = x13*x269*x6; 
      const auto x312 = x13*x271*x6; 
      const auto x313 = 4.0*x20 + x205; 
      const auto x314 = x10*x199; 
      const auto x315 = x10*x233; 
      const auto x316 = x10*x103; 
      const auto x317 = x10*x13*x269; 
      const auto x318 = x10*x13*x271; 
      const auto x319 = 6.0*x92; 
      const auto x320 = x13*x18*x269; 
      const auto x321 = x13*x18*x271; 
      const auto x322 = 12.0*x22 + x306*x51; 
      const auto x323 = x21*x274; 
      const auto x324 = x21*x269; 
      const auto x325 = x21*x271; 
      const auto x326 = 8.0*radial_eval_alpha_squared; 
      const auto x327 = 12.0*radial_eval_alpha; 
      const auto x328 = x16*x327; 
      const auto x329 = x105*x14; 
      const auto x330 = x14*x269; 
      const auto x331 = x14*x271; 
      const auto x332 = 3.0*x140; 
      const auto x333 = x10*x269*x6; 
      const auto x334 = x10*x271*x6; 
      const auto x335 = x326*x59; 
      const auto x336 = x240 + x335; 
      const auto x337 = x105*x18; 
      const auto x338 = x18*x269; 
      const auto x339 = x18*x271; 
      const auto x340 = x25*x274; 
      const auto x341 = x25*x269; 
      const auto x342 = x25*x271; 
      const auto x343 = x17*x327; 
      const auto x344 = 20.0*x67; 
      const auto x345 = 10.0*x273; 
      const auto x346 = x26*x269; 
      const auto x347 = x26*x271; 
      const auto x348 = 12.0*x67; 
      const auto x349 = 6.0*y; 
      const auto x350 = x10*x349*x67; 
      const auto x351 = x10*x14*x269; 
      const auto x352 = x10*x14*x271; 
      const auto x353 = 6.0*z; 
      const auto x354 = x353*x6*x67; 
      const auto x355 = x18*x269*x6; 
      const auto x356 = x18*x271*x6; 
      const auto x357 = x269*x29; 
      const auto x358 = x271*x29; 
      const auto x359 = 20.0*x70; 
      const auto x360 = radial_eval_alpha_squared*y; 
      const auto x361 = 10.0*x360; 
      const auto x362 = radial_eval_alpha_cubed*x13 + radial_eval_alpha_squared; 
      const auto x363 = x0*x362; 
      const auto x364 = radial_eval_alpha_cubed*x14 + 3.0*x360; 
      const auto x365 = x2*x364; 
      const auto x366 = x2*x362; 
      const auto x367 = 12.0*x70; 
      const auto x368 = 2.0*y; 
      const auto x369 = 6.0*x67; 
      const auto x370 = 18.0*x70; 
      const auto x371 = x362*x4*x6; 
      const auto x372 = x105*x4; 
      const auto x373 = x362*x4; 
      const auto x374 = x10*x362*x4; 
      const auto x375 = x13*x14*x362; 
      const auto x376 = x13*x362*x6; 
      const auto x377 = 4.0*x1 + x137; 
      const auto x378 = x10*x13*x362; 
      const auto x379 = x110*x13*x38; 
      const auto x380 = x13*x18*x362; 
      const auto x381 = x21*x362; 
      const auto x382 = 24.0*x1 + 4.0*x138 + x14*x275 + x21*x364 + 12.0*x218 + 4.0*x246 + x26*x276 + x325*y + x381*y; 
      const auto x383 = x14*x362; 
      const auto x384 = x10*x362*x6; 
      const auto x385 = x18*x362; 
      const auto x386 = x25*x362; 
      const auto x387 = x25*x364; 
      const auto x388 = x26*x362; 
      const auto x389 = x10*x14*x362; 
      const auto x390 = x23*x288 + x286 + x297 + x332; 
      const auto x391 = x18*x362*x6; 
      const auto x392 = x306*x88 + 12.0*x8; 
      const auto x393 = x23*x327; 
      const auto x394 = x29*x362; 
      const auto x395 = 20.0*x92; 
      const auto x396 = radial_eval_alpha_squared*z; 
      const auto x397 = 10.0*x396; 
      const auto x398 = radial_eval_alpha_cubed*x18 + 3.0*x396; 
      const auto x399 = 12.0*x92; 
      const auto x400 = x2*x398; 
      const auto x401 = 2.0*z; 
      const auto x402 = 18.0*x92; 
      const auto x403 = x141 + 4.0*x3; 
      const auto x404 = x21*x398; 
      const auto x405 = 4.0*x142 + x18*x275 + 4.0*x223 + 12.0*x249 + x25*x398 + x276*x29 + 24.0*x3 + x341*z + x386*z; 


      // Evaluate basis function
      basis_eval[ipt + 0*npts] = radial_eval*x0;
      basis_eval[ipt + 1*npts] = x1*x2;
      basis_eval[ipt + 2*npts] = x2*x3;
      basis_eval[ipt + 3*npts] = radial_eval*x7;
      basis_eval[ipt + 4*npts] = x4*x8;
      basis_eval[ipt + 5*npts] = radial_eval*x11;
      basis_eval[ipt + 6*npts] = radial_eval*x15;
      basis_eval[ipt + 7*npts] = x16*x3;
      basis_eval[ipt + 8*npts] = x1*x17;
      basis_eval[ipt + 9*npts] = radial_eval*x19;
      basis_eval[ipt + 10*npts] = x20*x21;
      basis_eval[ipt + 11*npts] = x14*x22;
      basis_eval[ipt + 12*npts] = x20*x23;
      basis_eval[ipt + 13*npts] = x18*x24;
      basis_eval[ipt + 14*npts] = x20*x25;
      basis_eval[ipt + 15*npts] = radial_eval*x26;
      basis_eval[ipt + 16*npts] = x21*x3;
      basis_eval[ipt + 17*npts] = radial_eval*x27;
      basis_eval[ipt + 18*npts] = radial_eval*x28;
      basis_eval[ipt + 19*npts] = x1*x25;
      basis_eval[ipt + 20*npts] = radial_eval*x29;


    
      // Evaluate first derivative of bfn wrt x
      basis_x_eval[ipt + 0*npts] = radial_eval_alpha*x30 + x2*x31;
      basis_x_eval[ipt + 1*npts] = x33*y;
      basis_x_eval[ipt + 2*npts] = x33*z;
      basis_x_eval[ipt + 3*npts] = x35 + x37;
      basis_x_eval[ipt + 4*npts] = x38*x40;
      basis_x_eval[ipt + 5*npts] = x42 + x43;
      basis_x_eval[ipt + 6*npts] = x14*x44 + x46;
      basis_x_eval[ipt + 7*npts] = x48*z;
      basis_x_eval[ipt + 8*npts] = x50*y;
      basis_x_eval[ipt + 9*npts] = x18*x44 + x52;
      basis_x_eval[ipt + 10*npts] = x53 + x55;
      basis_x_eval[ipt + 11*npts] = x58*z;
      basis_x_eval[ipt + 12*npts] = radial_eval*x23 + x60;
      basis_x_eval[ipt + 13*npts] = x63*y;
      basis_x_eval[ipt + 14*npts] = x64 + x66;
      basis_x_eval[ipt + 15*npts] = x26*x67;
      basis_x_eval[ipt + 16*npts] = x68;
      basis_x_eval[ipt + 17*npts] = x27*x67;
      basis_x_eval[ipt + 18*npts] = x28*x67;
      basis_x_eval[ipt + 19*npts] = x69;
      basis_x_eval[ipt + 20*npts] = x29*x67;

      // Evaluate first derivative of bfn wrt y
      basis_y_eval[ipt + 0*npts] = x0*x70;
      basis_y_eval[ipt + 1*npts] = x35 + x71;
      basis_y_eval[ipt + 2*npts] = x72;
      basis_y_eval[ipt + 3*npts] = x4*x73 + x46;
      basis_y_eval[ipt + 4*npts] = x75*z;
      basis_y_eval[ipt + 5*npts] = x11*x70;
      basis_y_eval[ipt + 6*npts] = x37 + x55;
      basis_y_eval[ipt + 7*npts] = x76*z;
      basis_y_eval[ipt + 8*npts] = radial_eval*x17 + x60;
      basis_y_eval[ipt + 9*npts] = x19*x70;
      basis_y_eval[ipt + 10*npts] = x*x77;
      basis_y_eval[ipt + 11*npts] = x78*x80;
      basis_y_eval[ipt + 12*npts] = x*x82;
      basis_y_eval[ipt + 13*npts] = x*(x61 + x83);
      basis_y_eval[ipt + 14*npts] = x69;
      basis_y_eval[ipt + 15*npts] = radial_eval_alpha*x84 + x21*x31;
      basis_y_eval[ipt + 16*npts] = x77*z;
      basis_y_eval[ipt + 17*npts] = x86 + x87;
      basis_y_eval[ipt + 18*npts] = x18*x73 + x89;
      basis_y_eval[ipt + 19*npts] = x64 + x91;
      basis_y_eval[ipt + 20*npts] = x29*x70;

      // Evaluate first derivative of bfn wrt z
      basis_z_eval[ipt + 0*npts] = x0*x92;
      basis_z_eval[ipt + 1*npts] = x72;
      basis_z_eval[ipt + 2*npts] = x42 + x71;
      basis_z_eval[ipt + 3*npts] = x7*x92;
      basis_z_eval[ipt + 4*npts] = y*(x49 + x74);
      basis_z_eval[ipt + 5*npts] = x4*x93 + x52;
      basis_z_eval[ipt + 6*npts] = x15*x92;
      basis_z_eval[ipt + 7*npts] = radial_eval*x16 + x60;
      basis_z_eval[ipt + 8*npts] = x94*y;
      basis_z_eval[ipt + 9*npts] = x43 + x66;
      basis_z_eval[ipt + 10*npts] = x68;
      basis_z_eval[ipt + 11*npts] = x*(x56 + x81);
      basis_z_eval[ipt + 12*npts] = x*x95;
      basis_z_eval[ipt + 13*npts] = x96*x98;
      basis_z_eval[ipt + 14*npts] = x*x99;
      basis_z_eval[ipt + 15*npts] = x26*x92;
      basis_z_eval[ipt + 16*npts] = x53 + x86;
      basis_z_eval[ipt + 17*npts] = x14*x93 + x89;
      basis_z_eval[ipt + 18*npts] = x87 + x91;
      basis_z_eval[ipt + 19*npts] = x99*y;
      basis_z_eval[ipt + 20*npts] = radial_eval_alpha*x100 + x25*x31;

      // Evaluate second derivative of bfn wrt xx
      basis_xx_eval[ipt + 0*npts] = x104;
      basis_xx_eval[ipt + 1*npts] = x109*y;
      basis_xx_eval[ipt + 2*npts] = x109*z;
      basis_xx_eval[ipt + 3*npts] = x110*x7 + x113;
      basis_xx_eval[ipt + 4*npts] = x38*(x110*x4 + x115);
      basis_xx_eval[ipt + 5*npts] = x11*x110 + x117;
      basis_xx_eval[ipt + 6*npts] = x119 + x122;
      basis_xx_eval[ipt + 7*npts] = z*(x123 + x126);
      basis_xx_eval[ipt + 8*npts] = y*(x127 + x130);
      basis_xx_eval[ipt + 9*npts] = x131 + x133;
      basis_xx_eval[ipt + 10*npts] = x*(x135 + x136);
      basis_xx_eval[ipt + 11*npts] = x78*(x137 + x138);
      basis_xx_eval[ipt + 12*npts] = x*(x139 + x140);
      basis_xx_eval[ipt + 13*npts] = x96*(x141 + x142);
      basis_xx_eval[ipt + 14*npts] = x*(x143 + x144);
      basis_xx_eval[ipt + 15*npts] = x145;
      basis_xx_eval[ipt + 16*npts] = x136*z;
      basis_xx_eval[ipt + 17*npts] = x146;
      basis_xx_eval[ipt + 18*npts] = x147;
      basis_xx_eval[ipt + 19*npts] = x144*y;
      basis_xx_eval[ipt + 20*npts] = x148;

      // Evaluate second derivative of bfn wrt xy
      basis_xy_eval[ipt + 0*npts] = x150*y;
      basis_xy_eval[ipt + 1*npts] = radial_eval_alpha_squared*x0*x6 + x151 + x33;
      basis_xy_eval[ipt + 2*npts] = x152;
      basis_xy_eval[ipt + 3*npts] = radial_eval_alpha_squared*x14*x2 + x15*x156 + x154 + x155*x2;
      basis_xy_eval[ipt + 4*npts] = z*(x159 + x40);
      basis_xy_eval[ipt + 5*npts] = x162*y;
      basis_xy_eval[ipt + 6*npts] = radial_eval_alpha_squared*x21*x4 + x112 + x156*x7 + x163*x21;
      basis_xy_eval[ipt + 7*npts] = z*(x164 + x167 + 4.0*x24);
      basis_xy_eval[ipt + 8*npts] = x169 + x50;
      basis_xy_eval[ipt + 9*npts] = x172*y;
      basis_xy_eval[ipt + 10*npts] = radial_eval_alpha_squared*x13*x26 + x119 + x77;
      basis_xy_eval[ipt + 11*npts] = z*(x174 + x80);
      basis_xy_eval[ipt + 12*npts] = x176 + x82;
      basis_xy_eval[ipt + 13*npts] = x177 + x63 + x83;
      basis_xy_eval[ipt + 14*npts] = y*(x178 + x97);
      basis_xy_eval[ipt + 15*npts] = x*x179;
      basis_xy_eval[ipt + 16*npts] = x180;
      basis_xy_eval[ipt + 17*npts] = x*x183;
      basis_xy_eval[ipt + 18*npts] = x*x186;
      basis_xy_eval[ipt + 19*npts] = x*(x187 + x97);
      basis_xy_eval[ipt + 20*npts] = radial_eval_alpha_squared*x29*x96;

      // Evaluate second derivative of bfn wrt xz
      basis_xz_eval[ipt + 0*npts] = x150*z;
      basis_xz_eval[ipt + 1*npts] = x152;
      basis_xz_eval[ipt + 2*npts] = radial_eval_alpha_squared*x0*x10 + x188 + x33;
      basis_xz_eval[ipt + 3*npts] = x159*z;
      basis_xz_eval[ipt + 4*npts] = y*(x162 + x40);
      basis_xz_eval[ipt + 5*npts] = radial_eval_alpha_squared*x18*x2 + x156*x19 + x190 + x191*x2;
      basis_xz_eval[ipt + 6*npts] = x167*z;
      basis_xz_eval[ipt + 7*npts] = x169 + x48;
      basis_xz_eval[ipt + 8*npts] = y*(x172 + x192 + 4.0*x22);
      basis_xz_eval[ipt + 9*npts] = radial_eval_alpha_squared*x25*x4 + x11*x156 + x116 + x163*x25;
      basis_xz_eval[ipt + 10*npts] = z*(x173 + x79);
      basis_xz_eval[ipt + 11*npts] = x175 + x58 + x81;
      basis_xz_eval[ipt + 12*npts] = x193 + x95;
      basis_xz_eval[ipt + 13*npts] = y*(x194 + x98);
      basis_xz_eval[ipt + 14*npts] = radial_eval_alpha_squared*x13*x29 + x131 + x99;
      basis_xz_eval[ipt + 15*npts] = radial_eval_alpha_squared*x26*x78;
      basis_xz_eval[ipt + 16*npts] = x*(x181 + x79);
      basis_xz_eval[ipt + 17*npts] = x*(x185 + x195);
      basis_xz_eval[ipt + 18*npts] = x*x196;
      basis_xz_eval[ipt + 19*npts] = x197;
      basis_xz_eval[ipt + 20*npts] = x*x198;

      // Evaluate second derivative of bfn wrt yy
      basis_yy_eval[ipt + 0*npts] = x200;
      basis_yy_eval[ipt + 1*npts] = y*(x201 + x202);
      basis_yy_eval[ipt + 2*npts] = x202*z;
      basis_yy_eval[ipt + 3*npts] = x151 + x204;
      basis_yy_eval[ipt + 4*npts] = x38*(x205 + x206);
      basis_yy_eval[ipt + 5*npts] = x207;
      basis_yy_eval[ipt + 6*npts] = x110*x15 + x208;
      basis_yy_eval[ipt + 7*npts] = z*(x123 + x211);
      basis_yy_eval[ipt + 8*npts] = y*(x212 + x213);
      basis_yy_eval[ipt + 9*npts] = x214;
      basis_yy_eval[ipt + 10*npts] = x*x217;
      basis_yy_eval[ipt + 11*npts] = x78*(x110*x14 + x219);
      basis_yy_eval[ipt + 12*npts] = x*(x220 + x222);
      basis_yy_eval[ipt + 13*npts] = x96*(x141 + x223);
      basis_yy_eval[ipt + 14*npts] = x*x224;
      basis_yy_eval[ipt + 15*npts] = x225;
      basis_yy_eval[ipt + 16*npts] = x217*z;
      basis_yy_eval[ipt + 17*npts] = x110*x27 + x227;
      basis_yy_eval[ipt + 18*npts] = x228 + x229;
      basis_yy_eval[ipt + 19*npts] = y*(x143 + x224);
      basis_yy_eval[ipt + 20*npts] = x230;

      // Evaluate second derivative of bfn wrt yz
      basis_yz_eval[ipt + 0*npts] = radial_eval_alpha_squared*x0*x38;
      basis_yz_eval[ipt + 1*npts] = z*(x157 + x39);
      basis_yz_eval[ipt + 2*npts] = y*(x160 + x39);
      basis_yz_eval[ipt + 3*npts] = z*(x164 + x166);
      basis_yz_eval[ipt + 4*npts] = x168 + x49 + x75;
      basis_yz_eval[ipt + 5*npts] = y*(x171 + x192);
      basis_yz_eval[ipt + 6*npts] = x174*z;
      basis_yz_eval[ipt + 7*npts] = x176 + x76;
      basis_yz_eval[ipt + 8*npts] = x193 + x94;
      basis_yz_eval[ipt + 9*npts] = x194*y;
      basis_yz_eval[ipt + 10*npts] = x180;
      basis_yz_eval[ipt + 11*npts] = x*(x183 + x80);
      basis_yz_eval[ipt + 12*npts] = x*(x186 + x195 + 4.0*x8);
      basis_yz_eval[ipt + 13*npts] = x*(x196 + x98);
      basis_yz_eval[ipt + 14*npts] = x197;
      basis_yz_eval[ipt + 15*npts] = x179*z;
      basis_yz_eval[ipt + 16*npts] = radial_eval_alpha_squared*x10*x26 + x231 + x77;
      basis_yz_eval[ipt + 17*npts] = radial_eval_alpha_squared*x18*x21 + x156*x28 + x191*x21 + x232;
      basis_yz_eval[ipt + 18*npts] = radial_eval_alpha_squared*x14*x25 + x155*x25 + x156*x27 + x226;
      basis_yz_eval[ipt + 19*npts] = radial_eval_alpha_squared*x29*x6 + x228 + x99;
      basis_yz_eval[ipt + 20*npts] = x198*y;

      // Evaluate second derivative of bfn wrt zz
      basis_zz_eval[ipt + 0*npts] = x234;
      basis_zz_eval[ipt + 1*npts] = x235*y;
      basis_zz_eval[ipt + 2*npts] = z*(x201 + x235);
      basis_zz_eval[ipt + 3*npts] = x236;
      basis_zz_eval[ipt + 4*npts] = x38*(x205 + x237);
      basis_zz_eval[ipt + 5*npts] = x188 + x238;
      basis_zz_eval[ipt + 6*npts] = x239;
      basis_zz_eval[ipt + 7*npts] = z*(x240 + x241);
      basis_zz_eval[ipt + 8*npts] = y*(x127 + x243);
      basis_zz_eval[ipt + 9*npts] = x110*x19 + x244;
      basis_zz_eval[ipt + 10*npts] = x*x245;
      basis_zz_eval[ipt + 11*npts] = x78*(x137 + x246);
      basis_zz_eval[ipt + 12*npts] = x*(x220 + x248);
      basis_zz_eval[ipt + 13*npts] = x96*(x110*x18 + x250);
      basis_zz_eval[ipt + 14*npts] = x*x253;
      basis_zz_eval[ipt + 15*npts] = x254;
      basis_zz_eval[ipt + 16*npts] = z*(x135 + x245);
      basis_zz_eval[ipt + 17*npts] = x231 + x255;
      basis_zz_eval[ipt + 18*npts] = x110*x28 + x256;
      basis_zz_eval[ipt + 19*npts] = x253*y;
      basis_zz_eval[ipt + 20*npts] = x257;

      // Evaluate Laplacian of bfn 
      basis_lapl_eval[ipt + 0*npts] = x104 + x200 + x234;
      basis_lapl_eval[ipt + 1*npts] = x258*y;
      basis_lapl_eval[ipt + 2*npts] = x258*z;
      basis_lapl_eval[ipt + 3*npts] = x102*x7 + x113 + x204 + x236;
      basis_lapl_eval[ipt + 4*npts] = x38*(x102*x4 + x206 + x259);
      basis_lapl_eval[ipt + 5*npts] = x102*x11 + x117 + x207 + x238;
      basis_lapl_eval[ipt + 6*npts] = x102*x15 + x122 + x208 + x239;
      basis_lapl_eval[ipt + 7*npts] = z*(x102*x16 + x241 + x260);
      basis_lapl_eval[ipt + 8*npts] = y*(x102*x17 + x213 + x261);
      basis_lapl_eval[ipt + 9*npts] = x102*x19 + x133 + x214 + x244;
      basis_lapl_eval[ipt + 10*npts] = x*x262;
      basis_lapl_eval[ipt + 11*npts] = x78*(x102*x14 + x138 + x263);
      basis_lapl_eval[ipt + 12*npts] = x*(x102*x23 + x140 + x264);
      basis_lapl_eval[ipt + 13*npts] = x96*(x102*x18 + x142 + x265);
      basis_lapl_eval[ipt + 14*npts] = x*x266;
      basis_lapl_eval[ipt + 15*npts] = x145 + x225 + x254;
      basis_lapl_eval[ipt + 16*npts] = x262*z;
      basis_lapl_eval[ipt + 17*npts] = x102*x27 + x146 + x227 + x255;
      basis_lapl_eval[ipt + 18*npts] = x102*x28 + x147 + x229 + x256;
      basis_lapl_eval[ipt + 19*npts] = x266*y;
      basis_lapl_eval[ipt + 20*npts] = x148 + x230 + x257;

      // Evaluate Laplacian gradient of bfn (dx)
      basis_lapl_x_eval[ipt + 0*npts] = x*x270 + x*x272 + x0*x274 + 15.0*x106 + x13*x267 + x2*x268 + 5.0*x202 + 5.0*x235;
      basis_lapl_x_eval[ipt + 1*npts] = x279*y;
      basis_lapl_x_eval[ipt + 2*npts] = x279*z;
      basis_lapl_x_eval[ipt + 3*npts] = x*x281 + x*x282 + 9.0*x125 + 3.0*x210 + x274*x4*x6 + x280*x34 + x285 + x289;
      basis_lapl_x_eval[ipt + 4*npts] = x38*(x*x293 + x*x294 + x13*x288 + x2*x280 + x274*x4 + x283 + 9.0*x290 + 3.0*x291 + 3.0*x292);
      basis_lapl_x_eval[ipt + 5*npts] = x*x295 + x*x296 + x10*x274*x4 + 9.0*x129 + 3.0*x242 + x280*x41 + x285 + x299;
      basis_lapl_x_eval[ipt + 6*npts] = x*x304 + x*x305 + x13*x14*x274 + x138*x302 + x14*x303 + x218*x300 + x246*x300 + x301*x4 + x307;
      basis_lapl_x_eval[ipt + 7*npts] = z*(x*x311 + x*x312 + x13*x274*x6 + x300*x308 + x300*x309 + x302*x310 + x303*x6 + x306*x7 + x313);
      basis_lapl_x_eval[ipt + 8*npts] = y*(x*x317 + x*x318 + x10*x13*x274 + x10*x303 + x11*x306 + x300*x314 + x300*x315 + x302*x316 + x313);
      basis_lapl_x_eval[ipt + 9*npts] = x*x320 + x*x321 + x13*x18*x274 + x142*x302 + x18*x303 + x223*x300 + x249*x300 + x319*x4 + x322;
      basis_lapl_x_eval[ipt + 10*npts] = x*x323 + x12*x324 + x12*x325 + 3.0*x136 + x217 + x245 + x326*x54 + x328;
      basis_lapl_x_eval[ipt + 11*npts] = z*(x*x14*x274 + x12*x330 + x12*x331 + x13*x301 + 3.0*x138 + x15*x326 + x263 + x329);
      basis_lapl_x_eval[ipt + 12*npts] = x*x10*x274*x6 + x105*x23 + x12*x333 + x12*x334 + x212 + x264 + x332 + x336;
      basis_lapl_x_eval[ipt + 13*npts] = y*(x*x18*x274 + x12*x338 + x12*x339 + x13*x319 + 3.0*x142 + x19*x326 + x265 + x337);
      basis_lapl_x_eval[ipt + 14*npts] = x*x340 + x12*x341 + x12*x342 + 3.0*x144 + x224 + x253 + x326*x65 + x343;
      basis_lapl_x_eval[ipt + 15*npts] = x*x346 + x*x347 + x14*x344 + x26*x274 + x26*x345;
      basis_lapl_x_eval[ipt + 16*npts] = z*(x*x324 + x*x325 + x21*x345 + x323 + x348*x6);
      basis_lapl_x_eval[ipt + 17*npts] = x*x351 + x*x352 + x10*x14*x274 + x165 + x27*x345 + x350;
      basis_lapl_x_eval[ipt + 18*npts] = x*x355 + x*x356 + x170 + x18*x274*x6 + x28*x345 + x354;
      basis_lapl_x_eval[ipt + 19*npts] = y*(x*x341 + x*x342 + x10*x348 + x25*x345 + x340);
      basis_lapl_x_eval[ipt + 20*npts] = x*x357 + x*x358 + x18*x344 + x274*x29 + x29*x345;
      // Evaluate Laplacian gradient of bfn (dy)
      basis_lapl_y_eval[ipt + 0*npts] = x0*x361 + x0*x364 + x272*y + x359*x4 + x363*y;
      basis_lapl_y_eval[ipt + 1*npts] = x109 + 3.0*x202 + x235 + x278*x5 + x326*x34 + x328 + x365*y + x366*x5;
      basis_lapl_y_eval[ipt + 2*npts] = z*(x13*x367 + x2*x361 + x278*y + x365 + x366*y);
      basis_lapl_y_eval[ipt + 3*npts] = x114*x368 + x14*x369 + x206*x349 + x237*x368 + x282*y + x307 + x364*x4*x6 + x370*x4 + x371*y;
      basis_lapl_y_eval[ipt + 4*npts] = z*(3.0*x206 + x259 + x294*x5 + x326*x7 + x364*x4*y + x369*x6 + x372 + x373*x5);
      basis_lapl_y_eval[ipt + 5*npts] = x10*x364*x4 + x11*x361 + x164 + x296*y + x350 + x374*y;
      basis_lapl_y_eval[ipt + 6*npts] = 3.0*x125 + x13*x14*x364 + x135 + 9.0*x210 + x280*x54 + x284 + x289 + x305*y + x375*y;
      basis_lapl_y_eval[ipt + 7*npts] = z*(x13*x364*x6 + x13*x370 + x15*x306 + x290*x368 + x291*x349 + x292*x368 + x312*y + x376*y + x377);
      basis_lapl_y_eval[ipt + 8*npts] = x10*x13*x364*y + x105*x17 + x139 + x261 + x298 + x318*x5 + x336 + x378*x5;
      basis_lapl_y_eval[ipt + 9*npts] = x13*x18*x364 + x184 + x19*x361 + x321*y + x379 + x380*y;
      basis_lapl_y_eval[ipt + 10*npts] = x*x382;
      basis_lapl_y_eval[ipt + 11*npts] = x78*(x14*x364 + x21*x280 + x283 + x288*x6 + 9.0*x308 + 3.0*x309 + 3.0*x310 + x331*y + x383*y);
      basis_lapl_y_eval[ipt + 12*npts] = x*(x10*x364*x6 + x10*x370 + x27*x306 + x314*x349 + x315*x368 + x316*x368 + x334*y + x377 + x384*y);
      basis_lapl_y_eval[ipt + 13*npts] = x*(x142 + x18*x364*y + 3.0*x223 + x250 + x28*x326 + x319*x6 + x337 + x339*x5 + x385*x5);
      basis_lapl_y_eval[ipt + 14*npts] = x*(x10*x367 + x25*x361 + x342*y + x386*y + x387);
      basis_lapl_y_eval[ipt + 15*npts] = 5.0*x136 + x21*x268 + 15.0*x215 + 5.0*x245 + x26*x364 + x267*x6 + x347*y + x388*y;
      basis_lapl_y_eval[ipt + 16*npts] = x382*z;
      basis_lapl_y_eval[ipt + 17*npts] = x10*x14*x364 + x135 + 9.0*x221 + 3.0*x247 + x280*x85 + x352*y + x389*y + x390;
      basis_lapl_y_eval[ipt + 18*npts] = x14*x319 + x142*x368 + x18*x364*x6 + x18*x370 + x223*x349 + x249*x368 + x356*y + x391*y + x392;
      basis_lapl_y_eval[ipt + 19*npts] = x144 + 3.0*x224 + x253 + x326*x90 + x342*x5 + x386*x5 + x387*y + x393;
      basis_lapl_y_eval[ipt + 20*npts] = x18*x359 + x29*x361 + x29*x364 + x358*y + x394*y;
      // Evaluate Laplacian gradient of bfn (dz)
      basis_lapl_z_eval[ipt + 0*npts] = x0*x397 + x0*x398 + x270*z + x363*z + x395*x4;
      basis_lapl_z_eval[ipt + 1*npts] = y*(x13*x399 + x2*x397 + x277*z + x366*z + x400);
      basis_lapl_z_eval[ipt + 2*npts] = x109 + x202 + 3.0*x235 + x277*x9 + x326*x41 + x343 + x366*x9 + x400*z;
      basis_lapl_z_eval[ipt + 3*npts] = x192 + x281*z + x354 + x371*z + x397*x7 + x398*x4*x6;
      basis_lapl_z_eval[ipt + 4*npts] = y*(x10*x369 + x11*x326 + x115 + x206 + 3.0*x237 + x293*x9 + x372 + x373*x9 + x398*x4*z);
      basis_lapl_z_eval[ipt + 5*npts] = x10*x398*x4 + x114*x401 + x18*x369 + x206*x401 + x237*x353 + x295*z + x322 + x374*z + x4*x402;
      basis_lapl_z_eval[ipt + 6*npts] = x13*x14*x398 + x15*x397 + x195 + x304*z + x375*z + x379;
      basis_lapl_z_eval[ipt + 7*npts] = x105*x16 + x13*x398*x6*z + x139 + x212 + x260 + x287 + x311*x9 + x335 + x376*x9;
      basis_lapl_z_eval[ipt + 8*npts] = y*(x10*x13*x398 + x13*x402 + x19*x306 + x290*x401 + x291*x401 + x292*x353 + x317*z + x378*z + x403);
      basis_lapl_z_eval[ipt + 9*npts] = 3.0*x129 + x13*x18*x398 + x143 + 9.0*x242 + x280*x65 + x284 + x299 + x320*z + x380*z;
      basis_lapl_z_eval[ipt + 10*npts] = x*(x21*x397 + x324*z + x381*z + x399*x6 + x404);
      basis_lapl_z_eval[ipt + 11*npts] = x*(x10*x301 + x138 + x14*x398*z + x219 + 3.0*x246 + x27*x326 + x329 + x330*x9 + x383*x9);
      basis_lapl_z_eval[ipt + 12*npts] = x*(x10*x398*x6 + x28*x306 + x308*x401 + x309*x353 + x310*x401 + x333*z + x384*z + x402*x6 + x403);
      basis_lapl_z_eval[ipt + 13*npts] = x96*(x10*x288 + x18*x398 + x25*x280 + x283 + 3.0*x314 + 9.0*x315 + 3.0*x316 + x338*z + x385*z);
      basis_lapl_z_eval[ipt + 14*npts] = x*x405;
      basis_lapl_z_eval[ipt + 15*npts] = x14*x395 + x26*x397 + x26*x398 + x346*z + x388*z;
      basis_lapl_z_eval[ipt + 16*npts] = x136 + x217 + 3.0*x245 + x324*x9 + x326*x85 + x381*x9 + x393 + x404*z;
      basis_lapl_z_eval[ipt + 17*npts] = x10*x14*x398 + x138*x401 + x14*x402 + x18*x301 + x218*x401 + x246*x353 + x351*z + x389*z + x392;
      basis_lapl_z_eval[ipt + 18*npts] = x143 + x18*x398*x6 + 3.0*x221 + 9.0*x247 + x280*x90 + x355*z + x390 + x391*z;
      basis_lapl_z_eval[ipt + 19*npts] = x405*y;
      basis_lapl_z_eval[ipt + 20*npts] = x10*x267 + 5.0*x144 + 5.0*x224 + x25*x268 + 15.0*x251 + x29*x398 + x357*z + x394*z;




#if 0
      // Evaluate the angular part of bfn



      double ang_eval_0;
      double ang_eval_1;
      double ang_eval_2;
      double ang_eval_3;


      ang_eval_0 = radial_eval*x0;
      ang_eval_1 = x1*x2;
      ang_eval_2 = x2*x3;
      ang_eval_3 = radial_eval*x7;
      basis_eval[ipt + 0*npts] = ang_eval_0;
      basis_eval[ipt + 1*npts] = ang_eval_1;
      basis_eval[ipt + 2*npts] = ang_eval_2;
      basis_eval[ipt + 3*npts] = ang_eval_3;

      ang_eval_0 = x4*x8;
      ang_eval_1 = radial_eval*x11;
      ang_eval_2 = radial_eval*x15;
      ang_eval_3 = x16*x3;
      basis_eval[ipt + 4*npts] = ang_eval_0;
      basis_eval[ipt + 5*npts] = ang_eval_1;
      basis_eval[ipt + 6*npts] = ang_eval_2;
      basis_eval[ipt + 7*npts] = ang_eval_3;

      ang_eval_0 = x1*x17;
      ang_eval_1 = radial_eval*x19;
      ang_eval_2 = x20*x21;
      ang_eval_3 = x14*x22;
      basis_eval[ipt + 8*npts] = ang_eval_0;
      basis_eval[ipt + 9*npts] = ang_eval_1;
      basis_eval[ipt + 10*npts] = ang_eval_2;
      basis_eval[ipt + 11*npts] = ang_eval_3;

      ang_eval_0 = x20*x23;
      ang_eval_1 = x18*x24;
      ang_eval_2 = x20*x25;
      ang_eval_3 = radial_eval*x26;
      basis_eval[ipt + 12*npts] = ang_eval_0;
      basis_eval[ipt + 13*npts] = ang_eval_1;
      basis_eval[ipt + 14*npts] = ang_eval_2;
      basis_eval[ipt + 15*npts] = ang_eval_3;

      ang_eval_0 = x21*x3;
      ang_eval_1 = radial_eval*x27;
      ang_eval_2 = radial_eval*x28;
      ang_eval_3 = x1*x25;
      basis_eval[ipt + 16*npts] = ang_eval_0;
      basis_eval[ipt + 17*npts] = ang_eval_1;
      basis_eval[ipt + 18*npts] = ang_eval_2;
      basis_eval[ipt + 19*npts] = ang_eval_3;

      ang_eval_0 = radial_eval*x29;
      basis_eval[ipt + 20*npts] = ang_eval_0;


      double dang_eval_x_0, dang_eval_y_0, dang_eval_z_0;
      double dang_eval_x_1, dang_eval_y_1, dang_eval_z_1;
      double dang_eval_x_2, dang_eval_y_2, dang_eval_z_2;
      double dang_eval_x_3, dang_eval_y_3, dang_eval_z_3;

      dang_eval_x_0 = radial_eval_alpha*x30 + x2*x31;
      dang_eval_y_0 = x0*x70;
      dang_eval_z_0 = x0*x92;
      dang_eval_x_1 = x33*y;
      dang_eval_y_1 = x35 + x71;
      dang_eval_z_1 = x72;
      dang_eval_x_2 = x33*z;
      dang_eval_y_2 = x72;
      dang_eval_z_2 = x42 + x71;
      dang_eval_x_3 = x35 + x37;
      dang_eval_y_3 = x4*x73 + x46;
      dang_eval_z_3 = x7*x92;
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

      dang_eval_x_0 = x38*x40;
      dang_eval_y_0 = x75*z;
      dang_eval_z_0 = y*(x49 + x74);
      dang_eval_x_1 = x42 + x43;
      dang_eval_y_1 = x11*x70;
      dang_eval_z_1 = x4*x93 + x52;
      dang_eval_x_2 = x14*x44 + x46;
      dang_eval_y_2 = x37 + x55;
      dang_eval_z_2 = x15*x92;
      dang_eval_x_3 = x48*z;
      dang_eval_y_3 = x76*z;
      dang_eval_z_3 = radial_eval*x16 + x60;
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

      dang_eval_x_0 = x50*y;
      dang_eval_y_0 = radial_eval*x17 + x60;
      dang_eval_z_0 = x94*y;
      dang_eval_x_1 = x18*x44 + x52;
      dang_eval_y_1 = x19*x70;
      dang_eval_z_1 = x43 + x66;
      dang_eval_x_2 = x53 + x55;
      dang_eval_y_2 = x*x77;
      dang_eval_z_2 = x68;
      dang_eval_x_3 = x58*z;
      dang_eval_y_3 = x78*x80;
      dang_eval_z_3 = x*(x56 + x81);
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

      dang_eval_x_0 = radial_eval*x23 + x60;
      dang_eval_y_0 = x*x82;
      dang_eval_z_0 = x*x95;
      dang_eval_x_1 = x63*y;
      dang_eval_y_1 = x*(x61 + x83);
      dang_eval_z_1 = x96*x98;
      dang_eval_x_2 = x64 + x66;
      dang_eval_y_2 = x69;
      dang_eval_z_2 = x*x99;
      dang_eval_x_3 = x26*x67;
      dang_eval_y_3 = radial_eval_alpha*x84 + x21*x31;
      dang_eval_z_3 = x26*x92;
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

      dang_eval_x_0 = x68;
      dang_eval_y_0 = x77*z;
      dang_eval_z_0 = x53 + x86;
      dang_eval_x_1 = x27*x67;
      dang_eval_y_1 = x86 + x87;
      dang_eval_z_1 = x14*x93 + x89;
      dang_eval_x_2 = x28*x67;
      dang_eval_y_2 = x18*x73 + x89;
      dang_eval_z_2 = x87 + x91;
      dang_eval_x_3 = x69;
      dang_eval_y_3 = x64 + x91;
      dang_eval_z_3 = x99*y;
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

      dang_eval_x_0 = x29*x67;
      dang_eval_y_0 = x29*x70;
      dang_eval_z_0 = radial_eval_alpha*x100 + x25*x31;
      basis_x_eval[ipt + 20*npts] = dang_eval_x_0;
      basis_y_eval[ipt + 20*npts] = dang_eval_y_0;
      basis_z_eval[ipt + 20*npts] = dang_eval_z_0;

#endif
    } // Loop over points within task
  } // Loop over tasks
        
  } // Loop over shells
} // end kernel

} // namespace GauXC
