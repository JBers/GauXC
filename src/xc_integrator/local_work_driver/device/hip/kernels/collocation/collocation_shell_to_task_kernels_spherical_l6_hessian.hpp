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


__global__ __launch_bounds__(128,2) void collocation_device_shell_to_task_kernel_spherical_hessian_6(
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
    auto* __restrict__ basis_xx_eval = task->d2bfxx + shoff;
    auto* __restrict__ basis_xy_eval = task->d2bfxy + shoff;
    auto* __restrict__ basis_xz_eval = task->d2bfxz + shoff;
    auto* __restrict__ basis_yy_eval = task->d2bfyy + shoff;
    auto* __restrict__ basis_yz_eval = task->d2bfyz + shoff;
    auto* __restrict__ basis_zz_eval = task->d2bfzz + shoff;

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
      const auto x0 = 0.0625*radial_eval; 
      const auto x1 = x*y; 
      const auto x2 = x0*x1; 
      const auto x3 = x*x; 
      const auto x4 = x3; 
      const auto x5 = y*y; 
      const auto x6 = x5; 
      const auto x7 = x4*x6; 
      const auto x8 = 10.0*x7; 
      const auto x9 = -x8; 
      const auto x10 = x*x*x*x; 
      const auto x11 = 3.0*x10; 
      const auto x12 = y*y*y*y; 
      const auto x13 = 3.0*x12; 
      const auto x14 = x11 + x13; 
      const auto x15 = x14 + x9; 
      const auto x16 = 0.1875*sqrt_154; 
      const auto x17 = y*z; 
      const auto x18 = x16*x17; 
      const auto x19 = 5.0*x10; 
      const auto x20 = x12 + x19; 
      const auto x21 = x20 + x9; 
      const auto x22 = 0.75*sqrt_7; 
      const auto x23 = x1*x22; 
      const auto x24 = z*z; 
      const auto x25 = x24; 
      const auto x26 = x25*x6; 
      const auto x27 = x25*x4; 
      const auto x28 = -x10; 
      const auto x29 = x12 + x28; 
      const auto x30 = -10.0*x26 + 10.0*x27 + x29; 
      const auto x31 = sqrt_210*x0; 
      const auto x32 = 9.0*x10; 
      const auto x33 = 24.0*x27; 
      const auto x34 = -x33; 
      const auto x35 = 8.0*x26; 
      const auto x36 = 6.0*x7; 
      const auto x37 = -x13 + x36; 
      const auto x38 = -x32 - x34 - x35 - x37; 
      const auto x39 = z*z*z*z; 
      const auto x40 = 16.0*x39; 
      const auto x41 = x10 + x12; 
      const auto x42 = -16.0*x26 - 16.0*x27 + x40 + x41; 
      const auto x43 = x42 + 2.0*x7; 
      const auto x44 = 0.125*sqrt_21; 
      const auto x45 = x17*x44; 
      const auto x46 = -20.0*x27; 
      const auto x47 = 5.0*x12; 
      const auto x48 = 8.0*x39; 
      const auto x49 = x19 - 20.0*x26 + x47 + x48 + x8; 
      const auto x50 = x46 + x49; 
      const auto x51 = x*x*x*x*x*x; 
      const auto x52 = y*y*y*y*y*y; 
      const auto x53 = x12*x4; 
      const auto x54 = 15.0*x53; 
      const auto x55 = x39*x4; 
      const auto x56 = x10*x6; 
      const auto x57 = 15.0*x56; 
      const auto x58 = x39*x6; 
      const auto x59 = 90.0*x10*x25 + 90.0*x12*x25 + 180.0*x25*x4*x6 - 5.0*x51 - 5.0*x52 - x54 - 120.0*x55 - x57 - 120.0*x58 + 16.0*(z*z*z*z*z*z); 
      const auto x60 = x*z; 
      const auto x61 = x44*x60; 
      const auto x62 = 0.03125*sqrt_210; 
      const auto x63 = x10*x25; 
      const auto x64 = x12*x25; 
      const auto x65 = -x52; 
      const auto x66 = x51 + x65; 
      const auto x67 = -x53 + 16.0*x55 + x56 - 16.0*x58 - 16.0*x63 + 16.0*x64 + x66; 
      const auto x68 = -24.0*x26; 
      const auto x69 = -x11 + x36; 
      const auto x70 = 9.0*x12 + x68 + x69; 
      const auto x71 = 8.0*x27 + x70; 
      const auto x72 = 0.1875*sqrt_7; 
      const auto x73 = -60.0*x25*x4*x6 - x51 + 5.0*x53 + 5.0*x56 + 10.0*x63 + 10.0*x64 + x65; 
      const auto x74 = x16*x60; 
      const auto x75 = x10 + x47; 
      const auto x76 = x75 + x9; 
      const auto x77 = 0.03125*sqrt_462; 
      const auto x78 = x54 - x57 + x66; 
      const auto x79 = 0.0625*sqrt_462; 
      const auto x80 = x79*y; 
      const auto x81 = x15*x4; 
      const auto x82 = 4.0*x; 
      const auto x83 = x*x*x; 
      const auto x84 = 3.0*x83; 
      const auto x85 = 5.0*x; 
      const auto x86 = -x6*x85; 
      const auto x87 = x84 + x86; 
      const auto x88 = x82*x87; 
      const auto x89 = x15 + x88; 
      const auto x90 = 20.0*radial_eval; 
      const auto x91 = x*x6; 
      const auto x92 = x83 - x91; 
      const auto x93 = radial_eval_alpha*x; 
      const auto x94 = x21*x93 + x90*x92; 
      const auto x95 = x22*y; 
      const auto x96 = x30*x4; 
      const auto x97 = x25*x85 - x83; 
      const auto x98 = x30 + x82*x97; 
      const auto x99 = 0.0625*sqrt_210; 
      const auto x100 = x17*x99; 
      const auto x101 = 12.0*radial_eval; 
      const auto x102 = -4.0*x*x25 + x84; 
      const auto x103 = -x102 - x91; 
      const auto x104 = x99*y; 
      const auto x105 = x4*x43; 
      const auto x106 = 8.0*x; 
      const auto x107 = x83 + x91; 
      const auto x108 = -x106*x25 + x107; 
      const auto x109 = x108*x82; 
      const auto x110 = x109 + x43; 
      const auto x111 = -2.0*x*x25; 
      const auto x112 = x107 + x111; 
      const auto x113 = 30.0*radial_eval; 
      const auto x114 = x*x*x*x*x; 
      const auto x115 = x*x12; 
      const auto x116 = x6*x83; 
      const auto x117 = 2.0*x116; 
      const auto x118 = 12.0*x*x25*x6 - x106*x39 - x114 - x115 - x117 + 12.0*x25*x83; 
      const auto x119 = x44*z; 
      const auto x120 = x4*x50; 
      const auto x121 = 20.0*x*x112; 
      const auto x122 = x121 + x50; 
      const auto x123 = 2.0*radial_eval; 
      const auto x124 = 3.0*x114; 
      const auto x125 = 16.0*x; 
      const auto x126 = x25*x83; 
      const auto x127 = -x115 + x117 + x124 + x125*x39 - 32.0*x126; 
      const auto x128 = x99*z; 
      const auto x129 = x4*x71; 
      const auto x130 = 3.0*x; 
      const auto x131 = x130*x6; 
      const auto x132 = x131 + x25*x82 - x84; 
      const auto x133 = x132*x82; 
      const auto x134 = x133 + x71; 
      const auto x135 = x12*x85; 
      const auto x136 = 10.0*x116; 
      const auto x137 = -60.0*x*x26 - x124 + 20.0*x126 + x135 + x136; 
      const auto x138 = x16*z; 
      const auto x139 = x4*x76; 
      const auto x140 = x83 + x86; 
      const auto x141 = x140*x82 + x76; 
      const auto x142 = radial_eval*x141 + radial_eval_alpha*x139; 
      const auto x143 = 6.0*radial_eval; 
      const auto x144 = x114 + x135 - x136; 
      const auto x145 = x*x79; 
      const auto x146 = x15*x6; 
      const auto x147 = 4.0*y; 
      const auto x148 = y*y*y; 
      const auto x149 = 3.0*x148; 
      const auto x150 = 5.0*y; 
      const auto x151 = -x150*x4; 
      const auto x152 = x149 + x151; 
      const auto x153 = x147*x152; 
      const auto x154 = x15 + x153; 
      const auto x155 = x21*x6; 
      const auto x156 = x148 + x151; 
      const auto x157 = x147*x156 + x21; 
      const auto x158 = radial_eval*x157 + radial_eval_alpha*x155; 
      const auto x159 = x*x22; 
      const auto x160 = x30*x6; 
      const auto x161 = x148 - x150*x25; 
      const auto x162 = x147*x161; 
      const auto x163 = x162 + x30; 
      const auto x164 = x38*x6; 
      const auto x165 = 3.0*y; 
      const auto x166 = x165*x4; 
      const auto x167 = x147*x25; 
      const auto x168 = 3.0*x148 - x166 - x167; 
      const auto x169 = x147*x168; 
      const auto x170 = -x36; 
      const auto x171 = -x32; 
      const auto x172 = x13 + x170 + x171 + x33 - x35; 
      const auto x173 = x169 + x172; 
      const auto x174 = x*x99; 
      const auto x175 = x43*x6; 
      const auto x176 = 8.0*y; 
      const auto x177 = x4*y; 
      const auto x178 = x148 + x177; 
      const auto x179 = -x176*x25 + x178; 
      const auto x180 = x147*x179; 
      const auto x181 = x180 + x43; 
      const auto x182 = x50*x6; 
      const auto x183 = 20.0*y; 
      const auto x184 = 2.0*y; 
      const auto x185 = -x184*x25; 
      const auto x186 = x178 + x185; 
      const auto x187 = x183*x186; 
      const auto x188 = x187 + x50; 
      const auto x189 = y*y*y*y*y; 
      const auto x190 = x10*y; 
      const auto x191 = x148*x4; 
      const auto x192 = 2.0*x191; 
      const auto x193 = 12.0*x148*x25 - x176*x39 - x189 - x190 - x192 + 12.0*x25*x4*y; 
      const auto x194 = radial_eval_alpha*y; 
      const auto x195 = 3.0*x189; 
      const auto x196 = 16.0*y; 
      const auto x197 = 32.0*x148*x25 + x190 - x192 - x195 - x196*x39; 
      const auto x198 = x60*x99; 
      const auto x199 = x149 - x167; 
      const auto x200 = x177 + x199; 
      const auto x201 = x10*x150; 
      const auto x202 = 20.0*x148*x25 + 10.0*x191 - x195 + x201 - 60.0*x27*y; 
      const auto x203 = x148 - x177; 
      const auto x204 = x194*x76 + x203*x90; 
      const auto x205 = 10.0*x148*x4 - x189 - x201; 
      const auto x206 = x17*x79; 
      const auto x207 = x16*y; 
      const auto x208 = x21*x25; 
      const auto x209 = -x6; 
      const auto x210 = x209 + x4; 
      const auto x211 = x210*x90; 
      const auto x212 = x25*x38; 
      const auto x213 = 16.0*x24; 
      const auto x214 = 3.0*x4; 
      const auto x215 = x209 + x214; 
      const auto x216 = x213*x215; 
      const auto x217 = x172 + x216; 
      const auto x218 = x1*x99; 
      const auto x219 = 32.0*radial_eval; 
      const auto x220 = x4*z; 
      const auto x221 = z*z*z; 
      const auto x222 = -2.0*x221; 
      const auto x223 = x222 + x6*z; 
      const auto x224 = -x220 - x223; 
      const auto x225 = radial_eval_alpha*z; 
      const auto x226 = x44*y; 
      const auto x227 = x25*x50; 
      const auto x228 = 5.0*z; 
      const auto x229 = 4.0*x221 - x228*x4 - x228*x6; 
      const auto x230 = 8.0*x229*z; 
      const auto x231 = x230 + x50; 
      const auto x232 = radial_eval*x231 + radial_eval_alpha*x227; 
      const auto x233 = 15.0*z; 
      const auto x234 = x221*x4; 
      const auto x235 = x221*x6; 
      const auto x236 = x10*x233 + x12*x233 - 40.0*x234 - 40.0*x235 + 30.0*x7*z + 8.0*(z*z*z*z*z); 
      const auto x237 = x*x44; 
      const auto x238 = -x10*z + x12*z + 2.0*x234 - 2.0*x235; 
      const auto x239 = x25*x71; 
      const auto x240 = 3.0*x6; 
      const auto x241 = -x240 + x4; 
      const auto x242 = x213*x241; 
      const auto x243 = x242 + x71; 
      const auto x244 = x72*z; 
      const auto x245 = x170 + x41; 
      const auto x246 = x245*x90; 
      const auto x247 = x*x16; 
      const auto x248 = x25*x76; 
      const auto x249 = radial_eval_alpha + radial_eval_alpha_squared*x4; 
      const auto x250 = x15*x249; 
      const auto x251 = 4.0*radial_eval; 
      const auto x252 = 9.0*x4; 
      const auto x253 = 5.0*x6; 
      const auto x254 = -x253; 
      const auto x255 = x*(x252 + x254) + 2.0*x87; 
      const auto x256 = 2.0*x93; 
      const auto x257 = x215*x90; 
      const auto x258 = 40.0*x93; 
      const auto x259 = x21*x249; 
      const auto x260 = 5.0*x25; 
      const auto x261 = -x*(x214 - x260) + 2.0*x97; 
      const auto x262 = x249*x30; 
      const auto x263 = 4.0*x25; 
      const auto x264 = -x263; 
      const auto x265 = -x252 - x264 - x6; 
      const auto x266 = 24.0*x93; 
      const auto x267 = x249*x38; 
      const auto x268 = -8.0*x25; 
      const auto x269 = x214 + x6; 
      const auto x270 = x*(x268 + x269) + 2.0*x108; 
      const auto x271 = x249*x43; 
      const auto x272 = -2.0*x25; 
      const auto x273 = x269 + x272; 
      const auto x274 = x249*x50; 
      const auto x275 = 12.0*x26; 
      const auto x276 = -x275; 
      const auto x277 = x36 + x48; 
      const auto x278 = -x113*(x20 - 36.0*x25*x4 + x276 + x277) + 60.0*x118*x93 + x249*x59; 
      const auto x279 = x*x273 + 2.0*x112; 
      const auto x280 = x36 + x40; 
      const auto x281 = radial_eval_alpha*x82; 
      const auto x282 = x123*(15.0*x10 - x12 - 96.0*x27 + x280) + x127*x281 + x249*x67; 
      const auto x283 = x*x249*x71 + x134*x256 + x251*(x*(x240 - x252 + x263) + 2.0*x132); 
      const auto x284 = 10.0*radial_eval; 
      const auto x285 = 12.0*x27; 
      const auto x286 = x137*x281 + x249*x73 + x284*(x12 + x276 + x285 + x69); 
      const auto x287 = x*x249*x76 + x141*x256 + x251*(x*(x214 + x254) + 2.0*x140); 
      const auto x288 = 12.0*x93; 
      const auto x289 = x113*x245 + x144*x288 + x249*x78; 
      const auto x290 = radial_eval_alpha*x3; 
      const auto x291 = radial_eval_alpha*x5; 
      const auto x292 = -x131 + x83; 
      const auto x293 = 20.0*radial_eval_alpha; 
      const auto x294 = radial_eval_alpha_squared*x; 
      const auto x295 = 12.0*radial_eval_alpha; 
      const auto x296 = 120.0*radial_eval; 
      const auto x297 = x*x148; 
      const auto x298 = x83*y; 
      const auto x299 = -6.0*x*x25*y + x297 + x298; 
      const auto x300 = x294*y; 
      const auto x301 = radial_eval_alpha_squared*y; 
      const auto x302 = 8.0*radial_eval; 
      const auto x303 = radial_eval_alpha*x184; 
      const auto x304 = x148 - x166; 
      const auto x305 = 3.0*z; 
      const auto x306 = 32.0*radial_eval_alpha; 
      const auto x307 = radial_eval_alpha_squared*z; 
      const auto x308 = 240.0*radial_eval; 
      const auto x309 = 30.0*x225; 
      const auto x310 = x294*z; 
      const auto x311 = radial_eval_alpha*x24; 
      const auto x312 = 128.0*radial_eval; 
      const auto x313 = 2.0*x225; 
      const auto x314 = 80.0*radial_eval; 
      const auto x315 = 2.0*radial_eval_alpha; 
      const auto x316 = x77*z; 
      const auto x317 = 6.0*radial_eval_alpha; 
      const auto x318 = radial_eval_alpha + radial_eval_alpha_squared*x6; 
      const auto x319 = x15*x318; 
      const auto x320 = 5.0*x4; 
      const auto x321 = -x320; 
      const auto x322 = 9.0*x6; 
      const auto x323 = 2.0*x152 + y*(x321 + x322); 
      const auto x324 = x157*x303 + x21*x318*y + x251*(2.0*x156 + y*(x240 + x321)); 
      const auto x325 = 2.0*x161 + y*(x240 - x260); 
      const auto x326 = x30*x318; 
      const auto x327 = x173*x303 + x251*(2.0*x168 - y*(x214 + x263 - 9.0*x6)) + x318*x38*y; 
      const auto x328 = x240 + x4; 
      const auto x329 = 2.0*x179 + y*(x268 + x328); 
      const auto x330 = x318*x43; 
      const auto x331 = x272 + x328; 
      const auto x332 = 2.0*x186 + x331*y; 
      const auto x333 = x318*x50; 
      const auto x334 = -x285; 
      const auto x335 = -x113*(-36.0*x25*x6 + x277 + x334 + x75) + 60.0*x193*x194 + x318*x59; 
      const auto x336 = 40.0*x194; 
      const auto x337 = radial_eval_alpha*x147; 
      const auto x338 = -x123*(15.0*x12 - 96.0*x25*x6 + x28 + x280) + x197*x337 + x318*x67; 
      const auto x339 = x264 + x322 + x4; 
      const auto x340 = x318*x71; 
      const auto x341 = x202*x337 + x284*(x10 + x275 + x334 + x37) + x318*x73; 
      const auto x342 = x240 - x4; 
      const auto x343 = x318*x76; 
      const auto x344 = 12.0*y; 
      const auto x345 = radial_eval_alpha*x344; 
      const auto x346 = -x113*x245 + x205*x345 + x318*x78; 
      const auto x347 = x301*z; 
      const auto x348 = radial_eval_alpha + radial_eval_alpha_squared*x25; 
      const auto x349 = x15*x348; 
      const auto x350 = x21*x348; 
      const auto x351 = 40.0*x311; 
      const auto x352 = x30*x348; 
      const auto x353 = 16.0*radial_eval; 
      const auto x354 = 3.0*x215; 
      const auto x355 = x348*x38; 
      const auto x356 = 6.0*x25 - x4 - x6; 
      const auto x357 = 64.0*x225; 
      const auto x358 = x348*x43; 
      const auto x359 = 2.0*x229 - z*(-12.0*x25 + x253 + x320); 
      const auto x360 = x348*x50; 
      const auto x361 = x231*x313 + x302*x359 + x360*z; 
      const auto x362 = 60.0*radial_eval*(x14 + x277 + x34 + x68) + 24.0*x225*x236 + x348*x59; 
      const auto x363 = x219*(-6.0*x26 + 6.0*x27 + x29) + x238*x357 + x348*x67; 
      const auto x364 = 3.0*x241; 
      const auto x365 = x348*x71; 
      const auto x366 = x245*x351 + x246 + x348*x73; 
      const auto x367 = x348*x76; 
      const auto x368 = x348*x78; 
      const auto x369 = radial_eval*x82; 
      const auto x370 = radial_eval*x147; 
      const auto x371 = x184*x93; 
      const auto x372 = x258*y; 
      const auto x373 = x266*y; 
      const auto x374 = x17*x258; 
      const auto x375 = x90*z; 
      const auto x376 = x184*x225; 
      const auto x377 = x256*z; 


      // Evaluate basis function
      basis_eval[ipt + 0*npts] = sqrt_462*x15*x2;
      basis_eval[ipt + 1*npts] = radial_eval*x18*x21;
      basis_eval[ipt + 2*npts] = radial_eval*x23*x30;
      basis_eval[ipt + 3*npts] = x17*x31*x38;
      basis_eval[ipt + 4*npts] = sqrt_210*x2*x43;
      basis_eval[ipt + 5*npts] = radial_eval*x45*x50;
      basis_eval[ipt + 6*npts] = x0*x59;
      basis_eval[ipt + 7*npts] = radial_eval*x50*x61;
      basis_eval[ipt + 8*npts] = radial_eval*x62*x67;
      basis_eval[ipt + 9*npts] = x31*x60*x71;
      basis_eval[ipt + 10*npts] = radial_eval*x72*x73;
      basis_eval[ipt + 11*npts] = radial_eval*x74*x76;
      basis_eval[ipt + 12*npts] = radial_eval*x77*x78;


    
      // Evaluate first derivative of bfn wrt x
      basis_x_eval[ipt + 0*npts] = x80*(radial_eval*x89 + radial_eval_alpha*x81);
      basis_x_eval[ipt + 1*npts] = x18*x94;
      basis_x_eval[ipt + 2*npts] = x95*(radial_eval*x98 + radial_eval_alpha*x96);
      basis_x_eval[ipt + 3*npts] = x100*(x101*x103 + x38*x93);
      basis_x_eval[ipt + 4*npts] = x104*(radial_eval*x110 + radial_eval_alpha*x105);
      basis_x_eval[ipt + 5*npts] = x45*(x112*x90 + x50*x93);
      basis_x_eval[ipt + 6*npts] = 0.0625*x113*x118 + 0.0625*x59*x93;
      basis_x_eval[ipt + 7*npts] = x119*(radial_eval*x122 + radial_eval_alpha*x120);
      basis_x_eval[ipt + 8*npts] = x62*(x123*x127 + x67*x93);
      basis_x_eval[ipt + 9*npts] = x128*(radial_eval*x134 + radial_eval_alpha*x129);
      basis_x_eval[ipt + 10*npts] = x72*(x123*x137 + x73*x93);
      basis_x_eval[ipt + 11*npts] = x138*x142;
      basis_x_eval[ipt + 12*npts] = x77*(x143*x144 + x78*x93);

      // Evaluate first derivative of bfn wrt y
      basis_y_eval[ipt + 0*npts] = x145*(radial_eval*x154 + radial_eval_alpha*x146);
      basis_y_eval[ipt + 1*npts] = x138*x158;
      basis_y_eval[ipt + 2*npts] = x159*(radial_eval*x163 + radial_eval_alpha*x160);
      basis_y_eval[ipt + 3*npts] = x128*(radial_eval*x173 + radial_eval_alpha*x164);
      basis_y_eval[ipt + 4*npts] = x174*(radial_eval*x181 + radial_eval_alpha*x175);
      basis_y_eval[ipt + 5*npts] = x119*(radial_eval*x188 + radial_eval_alpha*x182);
      basis_y_eval[ipt + 6*npts] = 0.0625*x113*x193 + 0.0625*x194*x59;
      basis_y_eval[ipt + 7*npts] = x61*(x186*x90 + x194*x50);
      basis_y_eval[ipt + 8*npts] = x62*(x123*x197 + x194*x67);
      basis_y_eval[ipt + 9*npts] = x198*(x101*x200 + x194*x71);
      basis_y_eval[ipt + 10*npts] = x72*(x123*x202 + x194*x73);
      basis_y_eval[ipt + 11*npts] = x204*x74;
      basis_y_eval[ipt + 12*npts] = x77*(x143*x205 + x194*x78);

      // Evaluate first derivative of bfn wrt z
      basis_z_eval[ipt + 0*npts] = x15*x206*x93;
      basis_z_eval[ipt + 1*npts] = x207*(radial_eval*x21 + radial_eval_alpha*x208);
      basis_z_eval[ipt + 2*npts] = x23*z*(radial_eval_alpha*x30 + x211);
      basis_z_eval[ipt + 3*npts] = x104*(radial_eval*x217 + radial_eval_alpha*x212);
      basis_z_eval[ipt + 4*npts] = x218*(x219*x224 + x225*x43);
      basis_z_eval[ipt + 5*npts] = x226*x232;
      basis_z_eval[ipt + 6*npts] = 0.0625*x101*x236 + 0.0625*x225*x59;
      basis_z_eval[ipt + 7*npts] = x232*x237;
      basis_z_eval[ipt + 8*npts] = x62*(x219*x238 + x225*x67);
      basis_z_eval[ipt + 9*npts] = x174*(radial_eval*x243 + radial_eval_alpha*x239);
      basis_z_eval[ipt + 10*npts] = x244*(radial_eval_alpha*x73 + x246);
      basis_z_eval[ipt + 11*npts] = x247*(radial_eval*x76 + radial_eval_alpha*x248);
      basis_z_eval[ipt + 12*npts] = x225*x77*x78;

      // Evaluate second derivative of bfn wrt xx
      basis_xx_eval[ipt + 0*npts] = x80*(x*x250 + x251*x255 + x256*x89);
      basis_xx_eval[ipt + 1*npts] = x18*(x257 + x258*x92 + x259);
      basis_xx_eval[ipt + 2*npts] = x95*(x*x262 + x251*x261 + x256*x98);
      basis_xx_eval[ipt + 3*npts] = x100*(x101*x265 + x103*x266 + x267);
      basis_xx_eval[ipt + 4*npts] = x104*(x*x271 + x110*x256 + x251*x270);
      basis_xx_eval[ipt + 5*npts] = x45*(x112*x258 + x273*x90 + x274);
      basis_xx_eval[ipt + 6*npts] = 0.0625*x278;
      basis_xx_eval[ipt + 7*npts] = x119*(x*x274 + x122*x256 + x279*x90);
      basis_xx_eval[ipt + 8*npts] = x282*x62;
      basis_xx_eval[ipt + 9*npts] = x128*x283;
      basis_xx_eval[ipt + 10*npts] = x286*x72;
      basis_xx_eval[ipt + 11*npts] = x138*x287;
      basis_xx_eval[ipt + 12*npts] = x289*x77;

      // Evaluate second derivative of bfn wrt xy
      basis_xy_eval[ipt + 0*npts] = x79*(radial_eval*(x14 + x153 - 50.0*x7 + x88) + radial_eval_alpha_squared*x15*x4*x6 + x154*x290 + x291*x89);
      basis_xy_eval[ipt + 1*npts] = x138*(x155*x294 + x157*x93 + x292*x90 + x293*x6*x92);
      basis_xy_eval[ipt + 2*npts] = x22*(radial_eval*(x162 + x98) + radial_eval_alpha_squared*x30*x4*x6 + x163*x290 + x291*x98);
      basis_xy_eval[ipt + 3*npts] = x128*(-x101*(x102 + x131) + x103*x295*x6 + x164*x294 + x173*x93);
      basis_xy_eval[ipt + 4*npts] = x99*(radial_eval*(x109 + x180 + x42 + x8) + radial_eval_alpha_squared*x4*x43*x6 + x110*x291 + x181*x290);
      basis_xy_eval[ipt + 5*npts] = x119*(x112*x293*x6 + x182*x294 + x188*x93 + x90*(x111 + x131 + x83));
      basis_xy_eval[ipt + 6*npts] = 1.875*x118*x194 + 1.875*x193*x93 - 0.0625*x296*x299 + 0.0625*x300*x59;
      basis_xy_eval[ipt + 7*npts] = x119*(x120*x301 + x122*x194 + x186*x293*x4 + x90*(x148 + x166 + x185));
      basis_xy_eval[ipt + 8*npts] = x62*(x127*x303 + x197*x256 + x300*x67 - x302*(x297 - x298));
      basis_xy_eval[ipt + 9*npts] = x128*(x101*(x166 + x199) + x129*x301 + x134*x194 + x200*x295*x4);
      basis_xy_eval[ipt + 10*npts] = x72*(40.0*radial_eval*x299 + x137*x303 + x202*x256 + x300*x73);
      basis_xy_eval[ipt + 11*npts] = x138*(x139*x301 + x141*x194 + x203*x293*x4 + x304*x90);
      basis_xy_eval[ipt + 12*npts] = x77*(6.0*x144*x194 + 6.0*x205*x93 + x296*(x297 - x298) + x300*x78);

      // Evaluate second derivative of bfn wrt xz
      basis_xz_eval[ipt + 0*npts] = x206*(radial_eval_alpha*x89 + radial_eval_alpha_squared*x81);
      basis_xz_eval[ipt + 1*npts] = x207*(x208*x294 + x25*x293*x92 + x94);
      basis_xz_eval[ipt + 2*npts] = x17*x22*(radial_eval_alpha*x98 + radial_eval_alpha_squared*x96 + x210*x293*x4 + x257);
      basis_xz_eval[ipt + 3*npts] = x104*(-x101*(-12.0*x*x25 + x84 + x91) + x103*x25*x295 + x212*x294 + x217*x93);
      basis_xz_eval[ipt + 4*npts] = x104*(x105*x307 + x110*x225 - x219*(x223 + x305*x4) + x224*x306*x4);
      basis_xz_eval[ipt + 5*npts] = x226*(x112*x25*x293 + x227*x294 + x231*x93 - x90*(6.0*x*x25 - x107));
      basis_xz_eval[ipt + 6*npts] = 0.0625*x118*x309 + 0.0625*x236*x288 + 0.0625*x308*(x130*x6*z - x221*x82 + x305*x83) + 0.0625*x310*x59;
      basis_xz_eval[ipt + 7*npts] = x44*(radial_eval*(x121 + x230 - 100.0*x27 + x49) + radial_eval_alpha_squared*x25*x4*x50 + x122*x311 + x231*x290);
      basis_xz_eval[ipt + 8*npts] = x62*(x127*x313 + 32.0*x238*x93 + x310*x67 + x312*(x*x221 - x83*z));
      basis_xz_eval[ipt + 9*npts] = x99*(radial_eval*(x133 + x242 + 40.0*x27 + x70) + radial_eval_alpha_squared*x25*x4*x71 + x134*x311 + x243*x290);
      basis_xz_eval[ipt + 10*npts] = x244*(x137*x315 + 20.0*x245*x93 + x292*x314 + x294*x73);
      basis_xz_eval[ipt + 11*npts] = x16*(radial_eval_alpha_squared*x25*x4*x76 + x141*x311 + x142);
      basis_xz_eval[ipt + 12*npts] = x316*(x144*x317 + x294*x78);

      // Evaluate second derivative of bfn wrt yy
      basis_yy_eval[ipt + 0*npts] = x145*(x154*x303 + x251*x323 + x319*y);
      basis_yy_eval[ipt + 1*npts] = x138*x324;
      basis_yy_eval[ipt + 2*npts] = x159*(x163*x303 + x251*x325 + x326*y);
      basis_yy_eval[ipt + 3*npts] = x128*x327;
      basis_yy_eval[ipt + 4*npts] = x174*(x181*x303 + x251*x329 + x330*y);
      basis_yy_eval[ipt + 5*npts] = x119*(x188*x303 + x332*x90 + x333*y);
      basis_yy_eval[ipt + 6*npts] = 0.0625*x335;
      basis_yy_eval[ipt + 7*npts] = x61*(x186*x336 + x331*x90 + x333);
      basis_yy_eval[ipt + 8*npts] = x338*x62;
      basis_yy_eval[ipt + 9*npts] = x198*(x101*x339 + 24.0*x194*x200 + x340);
      basis_yy_eval[ipt + 10*npts] = x341*x72;
      basis_yy_eval[ipt + 11*npts] = x74*(x203*x336 + x342*x90 + x343);
      basis_yy_eval[ipt + 12*npts] = x346*x77;

      // Evaluate second derivative of bfn wrt yz
      basis_yz_eval[ipt + 0*npts] = x60*x79*(radial_eval_alpha*x154 + radial_eval_alpha_squared*x146);
      basis_yz_eval[ipt + 1*npts] = x16*(radial_eval_alpha_squared*x21*x25*x6 + x157*x311 + x158);
      basis_yz_eval[ipt + 2*npts] = x22*x60*(radial_eval_alpha*x163 + radial_eval_alpha_squared*x160 + x210*x293*x6 + x241*x90);
      basis_yz_eval[ipt + 3*npts] = x99*(radial_eval*(x13 + x169 + x170 + x171 + x216 - 40.0*x26 + x33) + radial_eval_alpha_squared*x25*x38*x6 + x173*x311 + x217*x291);
      basis_yz_eval[ipt + 4*npts] = x174*(x175*x307 + x181*x225 - x219*(x220 + x222 + x305*x6) + x224*x306*x6);
      basis_yz_eval[ipt + 5*npts] = x44*(radial_eval*(x187 + x19 + x230 - 100.0*x26 + x46 + x47 + x48 + x8) + radial_eval_alpha_squared*x25*x50*x6 + x188*x311 + x231*x291);
      basis_yz_eval[ipt + 6*npts] = 0.0625*x193*x309 + 0.0625*x236*x345 + 0.0625*x308*(-x147*x221 + x148*x305 + x165*x4*z) + 0.0625*x347*x59;
      basis_yz_eval[ipt + 7*npts] = x237*(x186*x25*x293 + x194*x231 + x227*x301 + x90*(x178 - 6.0*x25*y));
      basis_yz_eval[ipt + 8*npts] = x62*(32.0*x194*x238 + x197*x313 + x312*(x148*z - x221*y) + x347*x67);
      basis_yz_eval[ipt + 9*npts] = x174*(x101*(x149 + x177 - x25*x344) + x194*x243 + x200*x25*x295 + x239*x301);
      basis_yz_eval[ipt + 10*npts] = x244*(radial_eval_alpha*x183*x245 + x202*x315 + x301*x73 + x304*x314);
      basis_yz_eval[ipt + 11*npts] = x247*(x203*x25*x293 + x204 + x248*x301);
      basis_yz_eval[ipt + 12*npts] = x316*(x205*x317 + x301*x78);

      // Evaluate second derivative of bfn wrt zz
      basis_zz_eval[ipt + 0*npts] = x1*x349*x79;
      basis_zz_eval[ipt + 1*npts] = x18*(x21*x315 + x350);
      basis_zz_eval[ipt + 2*npts] = x23*(x210*x351 + x211 + x352);
      basis_zz_eval[ipt + 3*npts] = x100*(x217*x315 + x353*x354 + x355);
      basis_zz_eval[ipt + 4*npts] = x218*(x219*x356 + x224*x357 + x358);
      basis_zz_eval[ipt + 5*npts] = x226*x361;
      basis_zz_eval[ipt + 6*npts] = 0.0625*x362;
      basis_zz_eval[ipt + 7*npts] = x237*x361;
      basis_zz_eval[ipt + 8*npts] = x363*x62;
      basis_zz_eval[ipt + 9*npts] = x198*(x243*x315 + x353*x364 + x365);
      basis_zz_eval[ipt + 10*npts] = x366*x72;
      basis_zz_eval[ipt + 11*npts] = x74*(x315*x76 + x367);
      basis_zz_eval[ipt + 12*npts] = x368*x77;






#if 0
      // Evaluate the angular part of bfn



      double ang_eval_0;
      double ang_eval_1;
      double ang_eval_2;
      double ang_eval_3;


      ang_eval_0 = sqrt_462*x15*x2;
      ang_eval_1 = radial_eval*x18*x21;
      ang_eval_2 = radial_eval*x23*x30;
      ang_eval_3 = x17*x31*x38;
      basis_eval[ipt + 0*npts] = ang_eval_0;
      basis_eval[ipt + 1*npts] = ang_eval_1;
      basis_eval[ipt + 2*npts] = ang_eval_2;
      basis_eval[ipt + 3*npts] = ang_eval_3;

      ang_eval_0 = sqrt_210*x2*x43;
      ang_eval_1 = radial_eval*x45*x50;
      ang_eval_2 = x0*x59;
      ang_eval_3 = radial_eval*x50*x61;
      basis_eval[ipt + 4*npts] = ang_eval_0;
      basis_eval[ipt + 5*npts] = ang_eval_1;
      basis_eval[ipt + 6*npts] = ang_eval_2;
      basis_eval[ipt + 7*npts] = ang_eval_3;

      ang_eval_0 = radial_eval*x62*x67;
      ang_eval_1 = x31*x60*x71;
      ang_eval_2 = radial_eval*x72*x73;
      ang_eval_3 = radial_eval*x74*x76;
      basis_eval[ipt + 8*npts] = ang_eval_0;
      basis_eval[ipt + 9*npts] = ang_eval_1;
      basis_eval[ipt + 10*npts] = ang_eval_2;
      basis_eval[ipt + 11*npts] = ang_eval_3;

      ang_eval_0 = radial_eval*x77*x78;
      basis_eval[ipt + 12*npts] = ang_eval_0;


      double dang_eval_x_0, dang_eval_y_0, dang_eval_z_0;
      double dang_eval_x_1, dang_eval_y_1, dang_eval_z_1;
      double dang_eval_x_2, dang_eval_y_2, dang_eval_z_2;
      double dang_eval_x_3, dang_eval_y_3, dang_eval_z_3;

      dang_eval_x_0 = x80*(radial_eval*x89 + radial_eval_alpha*x81);
      dang_eval_y_0 = x145*(radial_eval*x154 + radial_eval_alpha*x146);
      dang_eval_z_0 = x15*x206*x93;
      dang_eval_x_1 = x18*x94;
      dang_eval_y_1 = x138*x158;
      dang_eval_z_1 = x207*(radial_eval*x21 + radial_eval_alpha*x208);
      dang_eval_x_2 = x95*(radial_eval*x98 + radial_eval_alpha*x96);
      dang_eval_y_2 = x159*(radial_eval*x163 + radial_eval_alpha*x160);
      dang_eval_z_2 = x23*z*(radial_eval_alpha*x30 + x211);
      dang_eval_x_3 = x100*(x101*x103 + x38*x93);
      dang_eval_y_3 = x128*(radial_eval*x173 + radial_eval_alpha*x164);
      dang_eval_z_3 = x104*(radial_eval*x217 + radial_eval_alpha*x212);
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

      dang_eval_x_0 = x104*(radial_eval*x110 + radial_eval_alpha*x105);
      dang_eval_y_0 = x174*(radial_eval*x181 + radial_eval_alpha*x175);
      dang_eval_z_0 = x218*(x219*x224 + x225*x43);
      dang_eval_x_1 = x45*(x112*x90 + x50*x93);
      dang_eval_y_1 = x119*(radial_eval*x188 + radial_eval_alpha*x182);
      dang_eval_z_1 = x226*x232;
      dang_eval_x_2 = 0.0625*x113*x118 + 0.0625*x59*x93;
      dang_eval_y_2 = 0.0625*x113*x193 + 0.0625*x194*x59;
      dang_eval_z_2 = 0.0625*x101*x236 + 0.0625*x225*x59;
      dang_eval_x_3 = x119*(radial_eval*x122 + radial_eval_alpha*x120);
      dang_eval_y_3 = x61*(x186*x90 + x194*x50);
      dang_eval_z_3 = x232*x237;
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

      dang_eval_x_0 = x62*(x123*x127 + x67*x93);
      dang_eval_y_0 = x62*(x123*x197 + x194*x67);
      dang_eval_z_0 = x62*(x219*x238 + x225*x67);
      dang_eval_x_1 = x128*(radial_eval*x134 + radial_eval_alpha*x129);
      dang_eval_y_1 = x198*(x101*x200 + x194*x71);
      dang_eval_z_1 = x174*(radial_eval*x243 + radial_eval_alpha*x239);
      dang_eval_x_2 = x72*(x123*x137 + x73*x93);
      dang_eval_y_2 = x72*(x123*x202 + x194*x73);
      dang_eval_z_2 = x244*(radial_eval_alpha*x73 + x246);
      dang_eval_x_3 = x138*x142;
      dang_eval_y_3 = x204*x74;
      dang_eval_z_3 = x247*(radial_eval*x76 + radial_eval_alpha*x248);
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

      dang_eval_x_0 = x77*(x143*x144 + x78*x93);
      dang_eval_y_0 = x77*(x143*x205 + x194*x78);
      dang_eval_z_0 = x225*x77*x78;
      basis_x_eval[ipt + 12*npts] = dang_eval_x_0;
      basis_y_eval[ipt + 12*npts] = dang_eval_y_0;
      basis_z_eval[ipt + 12*npts] = dang_eval_z_0;

#endif
    } // Loop over points within task
  } // Loop over tasks
        
  } // Loop over shells
} // end kernel

} // namespace GauXC
