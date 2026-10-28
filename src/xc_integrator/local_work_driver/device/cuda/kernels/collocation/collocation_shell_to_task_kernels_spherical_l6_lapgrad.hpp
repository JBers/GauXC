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


__global__ __launch_bounds__(128,2) void collocation_device_shell_to_task_kernel_spherical_lapgrad_6(
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
      const auto x97 = -x83; 
      const auto x98 = x25*x85 + x97; 
      const auto x99 = x30 + x82*x98; 
      const auto x100 = 0.0625*sqrt_210; 
      const auto x101 = x100*x17; 
      const auto x102 = 12.0*radial_eval; 
      const auto x103 = x25*x82; 
      const auto x104 = -x103; 
      const auto x105 = x104 + x84; 
      const auto x106 = -x105 - x91; 
      const auto x107 = x100*y; 
      const auto x108 = x4*x43; 
      const auto x109 = 8.0*x; 
      const auto x110 = x83 + x91; 
      const auto x111 = -x109*x25 + x110; 
      const auto x112 = x111*x82; 
      const auto x113 = x112 + x43; 
      const auto x114 = 2.0*x; 
      const auto x115 = x114*x25; 
      const auto x116 = -x115; 
      const auto x117 = x110 + x116; 
      const auto x118 = 30.0*radial_eval; 
      const auto x119 = x*x*x*x*x; 
      const auto x120 = x*x12; 
      const auto x121 = x6*x83; 
      const auto x122 = 2.0*x121; 
      const auto x123 = 12.0*x*x25*x6 - x109*x39 - x119 - x120 - x122 + 12.0*x25*x83; 
      const auto x124 = x44*z; 
      const auto x125 = x4*x50; 
      const auto x126 = 20.0*x; 
      const auto x127 = x117*x126; 
      const auto x128 = x127 + x50; 
      const auto x129 = 2.0*radial_eval; 
      const auto x130 = 3.0*x119; 
      const auto x131 = 16.0*x; 
      const auto x132 = x25*x83; 
      const auto x133 = -x120 + x122 + x130 + x131*x39 - 32.0*x132; 
      const auto x134 = x100*z; 
      const auto x135 = x4*x71; 
      const auto x136 = 3.0*x; 
      const auto x137 = x136*x6; 
      const auto x138 = x103 + x137 - x84; 
      const auto x139 = x138*x82; 
      const auto x140 = x139 + x71; 
      const auto x141 = x12*x85; 
      const auto x142 = 10.0*x121; 
      const auto x143 = 60.0*x; 
      const auto x144 = -x130 + 20.0*x132 + x141 + x142 - x143*x26; 
      const auto x145 = x16*z; 
      const auto x146 = x4*x76; 
      const auto x147 = x83 + x86; 
      const auto x148 = x147*x82 + x76; 
      const auto x149 = radial_eval*x148 + radial_eval_alpha*x146; 
      const auto x150 = 6.0*radial_eval; 
      const auto x151 = x119 + x141 - x142; 
      const auto x152 = x*x79; 
      const auto x153 = x15*x6; 
      const auto x154 = 4.0*y; 
      const auto x155 = y*y*y; 
      const auto x156 = 3.0*x155; 
      const auto x157 = 5.0*y; 
      const auto x158 = -x157*x4; 
      const auto x159 = x156 + x158; 
      const auto x160 = x154*x159; 
      const auto x161 = x15 + x160; 
      const auto x162 = x21*x6; 
      const auto x163 = x155 + x158; 
      const auto x164 = x154*x163 + x21; 
      const auto x165 = radial_eval*x164 + radial_eval_alpha*x162; 
      const auto x166 = x*x22; 
      const auto x167 = x30*x6; 
      const auto x168 = x155 - x157*x25; 
      const auto x169 = x154*x168; 
      const auto x170 = x169 + x30; 
      const auto x171 = x38*x6; 
      const auto x172 = 3.0*y; 
      const auto x173 = x172*x4; 
      const auto x174 = x154*x25; 
      const auto x175 = 3.0*x155 - x173 - x174; 
      const auto x176 = x154*x175; 
      const auto x177 = -x36; 
      const auto x178 = -x32; 
      const auto x179 = x13 + x177 + x178 + x33 - x35; 
      const auto x180 = x176 + x179; 
      const auto x181 = x*x100; 
      const auto x182 = x43*x6; 
      const auto x183 = 8.0*y; 
      const auto x184 = x4*y; 
      const auto x185 = x155 + x184; 
      const auto x186 = -x183*x25 + x185; 
      const auto x187 = x154*x186; 
      const auto x188 = x187 + x43; 
      const auto x189 = x50*x6; 
      const auto x190 = 20.0*y; 
      const auto x191 = 2.0*y; 
      const auto x192 = x191*x25; 
      const auto x193 = -x192; 
      const auto x194 = x185 + x193; 
      const auto x195 = x190*x194; 
      const auto x196 = x195 + x50; 
      const auto x197 = y*y*y*y*y; 
      const auto x198 = x10*y; 
      const auto x199 = x155*x4; 
      const auto x200 = 2.0*x199; 
      const auto x201 = 12.0*x155*x25 - x183*x39 - x197 - x198 - x200 + 12.0*x25*x4*y; 
      const auto x202 = radial_eval_alpha*y; 
      const auto x203 = 3.0*x197; 
      const auto x204 = 16.0*y; 
      const auto x205 = 32.0*x155*x25 + x198 - x200 - x203 - x204*x39; 
      const auto x206 = x100*x60; 
      const auto x207 = -x174; 
      const auto x208 = x156 + x207; 
      const auto x209 = x184 + x208; 
      const auto x210 = x10*x157; 
      const auto x211 = 60.0*y; 
      const auto x212 = 20.0*x155*x25 + 10.0*x199 - x203 + x210 - x211*x27; 
      const auto x213 = x155 - x184; 
      const auto x214 = x202*x76 + x213*x90; 
      const auto x215 = 10.0*x155*x4 - x197 - x210; 
      const auto x216 = x17*x79; 
      const auto x217 = x16*y; 
      const auto x218 = x21*x25; 
      const auto x219 = x1*z; 
      const auto x220 = -x6; 
      const auto x221 = x220 + x4; 
      const auto x222 = x221*x90; 
      const auto x223 = x25*x38; 
      const auto x224 = 16.0*x24; 
      const auto x225 = 3.0*x4; 
      const auto x226 = x220 + x225; 
      const auto x227 = x224*x226; 
      const auto x228 = x179 + x227; 
      const auto x229 = x1*x100; 
      const auto x230 = 32.0*radial_eval; 
      const auto x231 = x4*z; 
      const auto x232 = z*z*z; 
      const auto x233 = -2.0*x232; 
      const auto x234 = x233 + x6*z; 
      const auto x235 = -x231 - x234; 
      const auto x236 = radial_eval_alpha*z; 
      const auto x237 = x44*y; 
      const auto x238 = x25*x50; 
      const auto x239 = -4.0*x232; 
      const auto x240 = 5.0*z; 
      const auto x241 = -x239 - x240*x4 - x240*x6; 
      const auto x242 = 8.0*x241*z; 
      const auto x243 = x242 + x50; 
      const auto x244 = radial_eval*x243 + radial_eval_alpha*x238; 
      const auto x245 = 15.0*z; 
      const auto x246 = x232*x4; 
      const auto x247 = x232*x6; 
      const auto x248 = 30.0*z; 
      const auto x249 = x10*x245 + x12*x245 - 40.0*x246 - 40.0*x247 + x248*x7 + 8.0*(z*z*z*z*z); 
      const auto x250 = x*x44; 
      const auto x251 = -x10*z + x12*z + 2.0*x246 - 2.0*x247; 
      const auto x252 = x25*x71; 
      const auto x253 = 3.0*x6; 
      const auto x254 = -x253; 
      const auto x255 = x254 + x4; 
      const auto x256 = x224*x255; 
      const auto x257 = x256 + x71; 
      const auto x258 = x72*z; 
      const auto x259 = x177 + x41; 
      const auto x260 = x259*x90; 
      const auto x261 = x*x16; 
      const auto x262 = x25*x76; 
      const auto x263 = radial_eval_alpha + radial_eval_alpha_squared*x4; 
      const auto x264 = x15*x263; 
      const auto x265 = 4.0*radial_eval; 
      const auto x266 = 9.0*x4; 
      const auto x267 = 5.0*x6; 
      const auto x268 = -x267; 
      const auto x269 = x*(x266 + x268) + 2.0*x87; 
      const auto x270 = 2.0*x93; 
      const auto x271 = x226*x90; 
      const auto x272 = 40.0*x93; 
      const auto x273 = x21*x263; 
      const auto x274 = -x225; 
      const auto x275 = 5.0*x25; 
      const auto x276 = x*(x274 + x275) + 2.0*x98; 
      const auto x277 = x265*x276; 
      const auto x278 = x263*x30; 
      const auto x279 = 4.0*x25; 
      const auto x280 = -x279; 
      const auto x281 = x266 + x280; 
      const auto x282 = -x281 - x6; 
      const auto x283 = 24.0*x93; 
      const auto x284 = x263*x38; 
      const auto x285 = -8.0*x25; 
      const auto x286 = x225 + x6; 
      const auto x287 = x*(x285 + x286) + 2.0*x111; 
      const auto x288 = x263*x43; 
      const auto x289 = -2.0*x25; 
      const auto x290 = x286 + x289; 
      const auto x291 = x263*x50; 
      const auto x292 = 12.0*x26; 
      const auto x293 = -x292; 
      const auto x294 = x36 + x48; 
      const auto x295 = -x20 + 36.0*x25*x4 - x293 - x294; 
      const auto x296 = 60.0*x93; 
      const auto x297 = x118*x295 + x123*x296 + x263*x59; 
      const auto x298 = x*x290 + 2.0*x117; 
      const auto x299 = x36 + x40; 
      const auto x300 = 15.0*x10 - x12 - 96.0*x27 + x299; 
      const auto x301 = radial_eval_alpha*x82; 
      const auto x302 = x129*x300 + x133*x301 + x263*x67; 
      const auto x303 = x253 + x279; 
      const auto x304 = -x*(x266 - x303) + 2.0*x138; 
      const auto x305 = x*x263*x71 + x140*x270 + x265*x304; 
      const auto x306 = 10.0*radial_eval; 
      const auto x307 = 12.0*x27; 
      const auto x308 = x12 + x293 + x307 + x69; 
      const auto x309 = x144*x301 + x263*x73 + x306*x308; 
      const auto x310 = x*(x225 + x268) + 2.0*x147; 
      const auto x311 = x*x263*x76 + x148*x270 + x265*x310; 
      const auto x312 = 12.0*x93; 
      const auto x313 = x118*x259 + x151*x312 + x263*x78; 
      const auto x314 = radial_eval_alpha*x3; 
      const auto x315 = radial_eval_alpha*x5; 
      const auto x316 = x14 + x160 - 50.0*x7 + x88; 
      const auto x317 = -x137 + x83; 
      const auto x318 = 20.0*radial_eval_alpha; 
      const auto x319 = x6*x92; 
      const auto x320 = radial_eval_alpha_squared*x; 
      const auto x321 = x169 + x99; 
      const auto x322 = -x105 - x137; 
      const auto x323 = 12.0*radial_eval_alpha; 
      const auto x324 = x106*x6; 
      const auto x325 = x112 + x187 + x42 + x8; 
      const auto x326 = x137 + x83; 
      const auto x327 = x116 + x326; 
      const auto x328 = x117*x6; 
      const auto x329 = 120.0*radial_eval; 
      const auto x330 = x*x155; 
      const auto x331 = x83*y; 
      const auto x332 = -6.0*x*x25*y + x330 + x331; 
      const auto x333 = -x332; 
      const auto x334 = 30.0*x93; 
      const auto x335 = 30.0*x202; 
      const auto x336 = x320*y; 
      const auto x337 = x155 + x173; 
      const auto x338 = x193 + x337; 
      const auto x339 = x194*x4; 
      const auto x340 = radial_eval_alpha_squared*y; 
      const auto x341 = 8.0*radial_eval; 
      const auto x342 = -x330 + x331; 
      const auto x343 = radial_eval_alpha*x191; 
      const auto x344 = x173 + x208; 
      const auto x345 = x209*x4; 
      const auto x346 = 40.0*radial_eval; 
      const auto x347 = x155 - x173; 
      const auto x348 = x213*x4; 
      const auto x349 = x330 - x331; 
      const auto x350 = 6.0*x202; 
      const auto x351 = 6.0*x93; 
      const auto x352 = x25*x92; 
      const auto x353 = x221*x4; 
      const auto x354 = 12.0*x; 
      const auto x355 = -x25*x354; 
      const auto x356 = -x355 - x84 - x91; 
      const auto x357 = x106*x25; 
      const auto x358 = 3.0*z; 
      const auto x359 = x358*x4; 
      const auto x360 = -x234 - x359; 
      const auto x361 = 32.0*radial_eval_alpha; 
      const auto x362 = x235*x4; 
      const auto x363 = radial_eval_alpha_squared*z; 
      const auto x364 = 6.0*x; 
      const auto x365 = -x25*x364; 
      const auto x366 = x110 + x365; 
      const auto x367 = x117*x25; 
      const auto x368 = 240.0*radial_eval; 
      const auto x369 = x136*x6*z - x232*x82 + x358*x83; 
      const auto x370 = 30.0*x236; 
      const auto x371 = x320*z; 
      const auto x372 = radial_eval_alpha*x24; 
      const auto x373 = x127 + x242 - 100.0*x27 + x49; 
      const auto x374 = 128.0*radial_eval; 
      const auto x375 = x*x232 - x83*z; 
      const auto x376 = 32.0*x93; 
      const auto x377 = 2.0*x236; 
      const auto x378 = x139 + x256 + 40.0*x27 + x70; 
      const auto x379 = 2.0*radial_eval_alpha; 
      const auto x380 = 80.0*radial_eval; 
      const auto x381 = 20.0*x93; 
      const auto x382 = x259*x381 + x317*x380; 
      const auto x383 = x77*z; 
      const auto x384 = 6.0*radial_eval_alpha; 
      const auto x385 = radial_eval_alpha + radial_eval_alpha_squared*x6; 
      const auto x386 = x15*x385; 
      const auto x387 = 5.0*x4; 
      const auto x388 = -x387; 
      const auto x389 = 9.0*x6; 
      const auto x390 = 2.0*x159 + y*(x388 + x389); 
      const auto x391 = 2.0*x163 + y*(x253 + x388); 
      const auto x392 = x164*x343 + x21*x385*y + x265*x391; 
      const auto x393 = -x275; 
      const auto x394 = 2.0*x168 + y*(x253 + x393); 
      const auto x395 = x265*x394; 
      const auto x396 = x30*x385; 
      const auto x397 = x225 + x279; 
      const auto x398 = 2.0*x175 - y*(x397 - 9.0*x6); 
      const auto x399 = x180*x343 + x265*x398 + x38*x385*y; 
      const auto x400 = x253 + x4; 
      const auto x401 = 2.0*x186 + y*(x285 + x400); 
      const auto x402 = x385*x43; 
      const auto x403 = x289 + x400; 
      const auto x404 = 2.0*x194 + x403*y; 
      const auto x405 = x385*x50; 
      const auto x406 = -x307; 
      const auto x407 = 36.0*x25*x6 - x294 - x406 - x75; 
      const auto x408 = 60.0*x202; 
      const auto x409 = x118*x407 + x201*x408 + x385*x59; 
      const auto x410 = 40.0*x202; 
      const auto x411 = -15.0*x12 + 96.0*x25*x6 - x28 - x299; 
      const auto x412 = radial_eval_alpha*x154; 
      const auto x413 = x129*x411 + x205*x412 + x385*x67; 
      const auto x414 = x280 + x389; 
      const auto x415 = x4 + x414; 
      const auto x416 = 24.0*x202; 
      const auto x417 = x385*x71; 
      const auto x418 = x10 + x292 + x37 + x406; 
      const auto x419 = x212*x412 + x306*x418 + x385*x73; 
      const auto x420 = -x4; 
      const auto x421 = x253 + x420; 
      const auto x422 = x385*x76; 
      const auto x423 = -x259; 
      const auto x424 = 12.0*y; 
      const auto x425 = radial_eval_alpha*x424; 
      const auto x426 = x118*x423 + x215*x425 + x385*x78; 
      const auto x427 = x221*x6; 
      const auto x428 = x13 + x176 + x177 + x178 + x227 - 40.0*x26 + x33; 
      const auto x429 = x358*x6; 
      const auto x430 = x233 + x429; 
      const auto x431 = -x231 - x430; 
      const auto x432 = x235*x6; 
      const auto x433 = x19 + x195 + x242 - 100.0*x26 + x46 + x47 + x48 + x8; 
      const auto x434 = -x154*x232 + x155*x358 + x172*x4*z; 
      const auto x435 = x340*z; 
      const auto x436 = 6.0*y; 
      const auto x437 = -x25*x436; 
      const auto x438 = x185 + x437; 
      const auto x439 = x194*x25; 
      const auto x440 = x155*z - x232*y; 
      const auto x441 = 32.0*x202; 
      const auto x442 = -x25*x424; 
      const auto x443 = x156 + x184 + x442; 
      const auto x444 = x209*x25; 
      const auto x445 = radial_eval_alpha*x190; 
      const auto x446 = x259*x445 + x347*x380; 
      const auto x447 = x213*x25; 
      const auto x448 = radial_eval_alpha + radial_eval_alpha_squared*x25; 
      const auto x449 = x15*x448; 
      const auto x450 = x21*x448; 
      const auto x451 = 40.0*x372; 
      const auto x452 = x30*x448; 
      const auto x453 = 16.0*radial_eval; 
      const auto x454 = 2.0*x226; 
      const auto x455 = x226 + x454; 
      const auto x456 = x38*x448; 
      const auto x457 = -6.0*x25; 
      const auto x458 = -x4 - x457 - x6; 
      const auto x459 = 64.0*x236; 
      const auto x460 = x43*x448; 
      const auto x461 = -12.0*x25; 
      const auto x462 = x267 + x387; 
      const auto x463 = 2.0*x241 - z*(x461 + x462); 
      const auto x464 = x448*x50; 
      const auto x465 = x243*x377 + x341*x463 + x464*z; 
      const auto x466 = 60.0*radial_eval; 
      const auto x467 = x14 + x294 + x34 + x68; 
      const auto x468 = 24.0*x236*x249 + x448*x59 + x466*x467; 
      const auto x469 = -6.0*x26 + 6.0*x27 + x29; 
      const auto x470 = x230*x469 + x251*x459 + x448*x67; 
      const auto x471 = 2.0*x255; 
      const auto x472 = x255 + x471; 
      const auto x473 = x448*x71; 
      const auto x474 = x259*x451 + x260 + x448*x73; 
      const auto x475 = x448*x76; 
      const auto x476 = x448*x78; 
      const auto x477 = radial_eval*x82; 
      const auto x478 = radial_eval*x154; 
      const auto x479 = x191*x93; 
      const auto x480 = x90*y; 
      const auto x481 = x226*x480; 
      const auto x482 = x272*y; 
      const auto x483 = x273*y + x392 + x481 + x482*x92; 
      const auto x484 = x24*y; 
      const auto x485 = x102*y; 
      const auto x486 = 24.0*y; 
      const auto x487 = x486*x93; 
      const auto x488 = 64.0*x93; 
      const auto x489 = x17*x272; 
      const auto x490 = x90*z; 
      const auto x491 = x191*x236; 
      const auto x492 = x270*z; 
      const auto x493 = x*x102; 
      const auto x494 = x*x90; 
      const auto x495 = x*x422 + x213*x482 + x311 + x421*x494; 
      const auto x496 = 15.0*x4; 
      const auto x497 = x3*y; 
      const auto x498 = radial_eval_alpha_cubed*x6 + radial_eval_alpha_squared; 
      const auto x499 = x15*x498; 
      const auto x500 = radial_eval_alpha_cubed*x25 + radial_eval_alpha_squared; 
      const auto x501 = x15*x500; 
      const auto x502 = x424*x93; 
      const auto x503 = radial_eval_alpha_cubed*x83 + radial_eval_alpha_squared*x136; 
      const auto x504 = 4.0*x314; 
      const auto x505 = radial_eval_alpha_squared*x191; 
      const auto x506 = x3*x505; 
      const auto x507 = x211*x93; 
      const auto x508 = x114*x340; 
      const auto x509 = x21*x498; 
      const auto x510 = x21*x500; 
      const auto x511 = 40.0*x340; 
      const auto x512 = x24*x511; 
      const auto x513 = x30*x498; 
      const auto x514 = x30*x500; 
      const auto x515 = 36.0*y; 
      const auto x516 = x515*x93; 
      const auto x517 = x204*x93; 
      const auto x518 = x38*x498; 
      const auto x519 = x38*x500; 
      const auto x520 = x387 + x6; 
      const auto x521 = x286 + x457; 
      const auto x522 = 64.0*z; 
      const auto x523 = 64.0*x435; 
      const auto x524 = x43*x498; 
      const auto x525 = x43*x500; 
      const auto x526 = x50*x503; 
      const auto x527 = 20.0*z; 
      const auto x528 = x190*z; 
      const auto x529 = x410*z; 
      const auto x530 = x211*z; 
      const auto x531 = x498*x50; 
      const auto x532 = x114*x435; 
      const auto x533 = x530*x93; 
      const auto x534 = x50*x500; 
      const auto x535 = x183*x463*x93 + x219*x534 + x243*x532; 
      const auto x536 = 720.0*radial_eval; 
      const auto x537 = x326 + x365; 
      const auto x538 = 5.0*x83; 
      const auto x539 = 240.0*x202; 
      const auto x540 = 480.0*x236; 
      const auto x541 = 90.0*x93; 
      const auto x542 = x143*x340; 
      const auto x543 = x498*x59; 
      const auto x544 = x500*x59; 
      const auto x545 = x466*z; 
      const auto x546 = x490*(x225 + x253 + x289); 
      const auto x547 = 20.0*x236; 
      const auto x548 = 60.0*z; 
      const auto x549 = x3*z; 
      const auto x550 = 2.0*x363; 
      const auto x551 = 256.0*x236; 
      const auto x552 = 24.0*radial_eval; 
      const auto x553 = 64.0*x371; 
      const auto x554 = x340*x82; 
      const auto x555 = x498*x67; 
      const auto x556 = x500*x67; 
      const auto x557 = x498*x71; 
      const auto x558 = x500*x71; 
      const auto x559 = 2.0*radial_eval_alpha_squared; 
      const auto x560 = x498*x73; 
      const auto x561 = x500*x73; 
      const auto x562 = 160.0*x372; 
      const auto x563 = 40.0*x320; 
      const auto x564 = x24*x563; 
      const auto x565 = x498*x76; 
      const auto x566 = x500*x76; 
      const auto x567 = x340*x354; 
      const auto x568 = x498*x78; 
      const auto x569 = x500*x78; 
      const auto x570 = x*x5; 
      const auto x571 = radial_eval_alpha_cubed*x4 + radial_eval_alpha_squared; 
      const auto x572 = x15*x571; 
      const auto x573 = radial_eval_alpha_cubed*x155 + radial_eval_alpha_squared*x172; 
      const auto x574 = 4.0*x315; 
      const auto x575 = radial_eval_alpha_squared*x114; 
      const auto x576 = x5*x575; 
      const auto x577 = x21*x571; 
      const auto x578 = x30*x571; 
      const auto x579 = x38*x571; 
      const auto x580 = x267 + x4; 
      const auto x581 = x400 + x457; 
      const auto x582 = x43*x571; 
      const auto x583 = x272*z; 
      const auto x584 = x5*z; 
      const auto x585 = x50*x571; 
      const auto x586 = x50*x573; 
      const auto x587 = x337 + x437; 
      const auto x588 = 5.0*x155; 
      const auto x589 = 240.0*x93; 
      const auto x590 = 90.0*x202; 
      const auto x591 = x571*x59; 
      const auto x592 = x126*z; 
      const auto x593 = -x155; 
      const auto x594 = x173 + x593; 
      const auto x595 = x571*x67; 
      const auto x596 = x571*x71; 
      const auto x597 = x571*x73; 
      const auto x598 = x571*x76; 
      const auto x599 = x571*x78; 
      const auto x600 = x236*x82; 
      const auto x601 = x154*x236; 
      const auto x602 = radial_eval_alpha_cubed*x232 + radial_eval_alpha_squared*x358; 
      const auto x603 = 4.0*x372; 
      const auto x604 = x24*x505; 
      const auto x605 = 40.0*x336; 
      const auto x606 = x126*x17; 
      const auto x607 = 48.0*radial_eval; 
      const auto x608 = x320*x486; 
      const auto x609 = 48.0*x24; 
      const auto x610 = x488*y; 
      const auto x611 = 32.0*x1; 
      const auto x612 = 20.0*x25 - x462; 
      const auto x613 = x24*x318; 
      const auto x614 = x50*x602; 
      const auto x615 = x243*x263; 
      const auto x616 = x243*x385; 
      const auto x617 = x243*x448; 
      const auto x618 = 9.0*z; 
      const auto x619 = x*x24; 
      const auto x620 = x24*x575; 
      const auto x621 = x154*x363; 
      const auto x622 = x363*x82; 
      const auto x623 = x368*z; 
      const auto x624 = 160.0*z; 
      const auto x625 = 10.0*x236; 


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
      basis_x_eval[ipt + 2*npts] = x95*(radial_eval*x99 + radial_eval_alpha*x96);
      basis_x_eval[ipt + 3*npts] = x101*(x102*x106 + x38*x93);
      basis_x_eval[ipt + 4*npts] = x107*(radial_eval*x113 + radial_eval_alpha*x108);
      basis_x_eval[ipt + 5*npts] = x45*(x117*x90 + x50*x93);
      basis_x_eval[ipt + 6*npts] = 0.0625*x118*x123 + 0.0625*x59*x93;
      basis_x_eval[ipt + 7*npts] = x124*(radial_eval*x128 + radial_eval_alpha*x125);
      basis_x_eval[ipt + 8*npts] = x62*(x129*x133 + x67*x93);
      basis_x_eval[ipt + 9*npts] = x134*(radial_eval*x140 + radial_eval_alpha*x135);
      basis_x_eval[ipt + 10*npts] = x72*(x129*x144 + x73*x93);
      basis_x_eval[ipt + 11*npts] = x145*x149;
      basis_x_eval[ipt + 12*npts] = x77*(x150*x151 + x78*x93);

      // Evaluate first derivative of bfn wrt y
      basis_y_eval[ipt + 0*npts] = x152*(radial_eval*x161 + radial_eval_alpha*x153);
      basis_y_eval[ipt + 1*npts] = x145*x165;
      basis_y_eval[ipt + 2*npts] = x166*(radial_eval*x170 + radial_eval_alpha*x167);
      basis_y_eval[ipt + 3*npts] = x134*(radial_eval*x180 + radial_eval_alpha*x171);
      basis_y_eval[ipt + 4*npts] = x181*(radial_eval*x188 + radial_eval_alpha*x182);
      basis_y_eval[ipt + 5*npts] = x124*(radial_eval*x196 + radial_eval_alpha*x189);
      basis_y_eval[ipt + 6*npts] = 0.0625*x118*x201 + 0.0625*x202*x59;
      basis_y_eval[ipt + 7*npts] = x61*(x194*x90 + x202*x50);
      basis_y_eval[ipt + 8*npts] = x62*(x129*x205 + x202*x67);
      basis_y_eval[ipt + 9*npts] = x206*(x102*x209 + x202*x71);
      basis_y_eval[ipt + 10*npts] = x72*(x129*x212 + x202*x73);
      basis_y_eval[ipt + 11*npts] = x214*x74;
      basis_y_eval[ipt + 12*npts] = x77*(x150*x215 + x202*x78);

      // Evaluate first derivative of bfn wrt z
      basis_z_eval[ipt + 0*npts] = x15*x216*x93;
      basis_z_eval[ipt + 1*npts] = x217*(radial_eval*x21 + radial_eval_alpha*x218);
      basis_z_eval[ipt + 2*npts] = x219*x22*(radial_eval_alpha*x30 + x222);
      basis_z_eval[ipt + 3*npts] = x107*(radial_eval*x228 + radial_eval_alpha*x223);
      basis_z_eval[ipt + 4*npts] = x229*(x230*x235 + x236*x43);
      basis_z_eval[ipt + 5*npts] = x237*x244;
      basis_z_eval[ipt + 6*npts] = 0.0625*x102*x249 + 0.0625*x236*x59;
      basis_z_eval[ipt + 7*npts] = x244*x250;
      basis_z_eval[ipt + 8*npts] = x62*(x230*x251 + x236*x67);
      basis_z_eval[ipt + 9*npts] = x181*(radial_eval*x257 + radial_eval_alpha*x252);
      basis_z_eval[ipt + 10*npts] = x258*(radial_eval_alpha*x73 + x260);
      basis_z_eval[ipt + 11*npts] = x261*(radial_eval*x76 + radial_eval_alpha*x262);
      basis_z_eval[ipt + 12*npts] = x236*x77*x78;

      // Evaluate second derivative of bfn wrt xx
      basis_xx_eval[ipt + 0*npts] = x80*(x*x264 + x265*x269 + x270*x89);
      basis_xx_eval[ipt + 1*npts] = x18*(x271 + x272*x92 + x273);
      basis_xx_eval[ipt + 2*npts] = x95*(x*x278 + x270*x99 + x277);
      basis_xx_eval[ipt + 3*npts] = x101*(x102*x282 + x106*x283 + x284);
      basis_xx_eval[ipt + 4*npts] = x107*(x*x288 + x113*x270 + x265*x287);
      basis_xx_eval[ipt + 5*npts] = x45*(x117*x272 + x290*x90 + x291);
      basis_xx_eval[ipt + 6*npts] = 0.0625*x297;
      basis_xx_eval[ipt + 7*npts] = x124*(x*x291 + x128*x270 + x298*x90);
      basis_xx_eval[ipt + 8*npts] = x302*x62;
      basis_xx_eval[ipt + 9*npts] = x134*x305;
      basis_xx_eval[ipt + 10*npts] = x309*x72;
      basis_xx_eval[ipt + 11*npts] = x145*x311;
      basis_xx_eval[ipt + 12*npts] = x313*x77;

      // Evaluate second derivative of bfn wrt xy
      basis_xy_eval[ipt + 0*npts] = x79*(radial_eval*x316 + radial_eval_alpha_squared*x15*x4*x6 + x161*x314 + x315*x89);
      basis_xy_eval[ipt + 1*npts] = x145*(x162*x320 + x164*x93 + x317*x90 + x318*x319);
      basis_xy_eval[ipt + 2*npts] = x22*(radial_eval*x321 + radial_eval_alpha_squared*x30*x4*x6 + x170*x314 + x315*x99);
      basis_xy_eval[ipt + 3*npts] = x134*(x102*x322 + x171*x320 + x180*x93 + x323*x324);
      basis_xy_eval[ipt + 4*npts] = x100*(radial_eval*x325 + radial_eval_alpha_squared*x4*x43*x6 + x113*x315 + x188*x314);
      basis_xy_eval[ipt + 5*npts] = x124*(x189*x320 + x196*x93 + x318*x328 + x327*x90);
      basis_xy_eval[ipt + 6*npts] = 0.0625*x123*x335 + 0.0625*x201*x334 + 0.0625*x329*x333 + 0.0625*x336*x59;
      basis_xy_eval[ipt + 7*npts] = x124*(x125*x340 + x128*x202 + x318*x339 + x338*x90);
      basis_xy_eval[ipt + 8*npts] = x62*(x133*x343 + x205*x270 + x336*x67 + x341*x342);
      basis_xy_eval[ipt + 9*npts] = x134*(x102*x344 + x135*x340 + x140*x202 + x323*x345);
      basis_xy_eval[ipt + 10*npts] = x72*(x144*x343 + x212*x270 + x332*x346 + x336*x73);
      basis_xy_eval[ipt + 11*npts] = x145*(x146*x340 + x148*x202 + x318*x348 + x347*x90);
      basis_xy_eval[ipt + 12*npts] = x77*(x151*x350 + x215*x351 + x329*x349 + x336*x78);

      // Evaluate second derivative of bfn wrt xz
      basis_xz_eval[ipt + 0*npts] = x216*(radial_eval_alpha*x89 + radial_eval_alpha_squared*x81);
      basis_xz_eval[ipt + 1*npts] = x217*(x218*x320 + x318*x352 + x94);
      basis_xz_eval[ipt + 2*npts] = x17*x22*(radial_eval_alpha*x99 + radial_eval_alpha_squared*x96 + x271 + x318*x353);
      basis_xz_eval[ipt + 3*npts] = x107*(x102*x356 + x223*x320 + x228*x93 + x323*x357);
      basis_xz_eval[ipt + 4*npts] = x107*(x108*x363 + x113*x236 + x230*x360 + x361*x362);
      basis_xz_eval[ipt + 5*npts] = x237*(x238*x320 + x243*x93 + x318*x367 + x366*x90);
      basis_xz_eval[ipt + 6*npts] = 0.0625*x123*x370 + 0.0625*x249*x312 + 0.0625*x368*x369 + 0.0625*x371*x59;
      basis_xz_eval[ipt + 7*npts] = x44*(radial_eval*x373 + radial_eval_alpha_squared*x25*x4*x50 + x128*x372 + x243*x314);
      basis_xz_eval[ipt + 8*npts] = x62*(x133*x377 + x251*x376 + x371*x67 + x374*x375);
      basis_xz_eval[ipt + 9*npts] = x100*(radial_eval*x378 + radial_eval_alpha_squared*x25*x4*x71 + x140*x372 + x257*x314);
      basis_xz_eval[ipt + 10*npts] = x258*(x144*x379 + x320*x73 + x382);
      basis_xz_eval[ipt + 11*npts] = x16*(radial_eval_alpha_squared*x25*x4*x76 + x148*x372 + x149);
      basis_xz_eval[ipt + 12*npts] = x383*(x151*x384 + x320*x78);

      // Evaluate second derivative of bfn wrt yy
      basis_yy_eval[ipt + 0*npts] = x152*(x161*x343 + x265*x390 + x386*y);
      basis_yy_eval[ipt + 1*npts] = x145*x392;
      basis_yy_eval[ipt + 2*npts] = x166*(x170*x343 + x395 + x396*y);
      basis_yy_eval[ipt + 3*npts] = x134*x399;
      basis_yy_eval[ipt + 4*npts] = x181*(x188*x343 + x265*x401 + x402*y);
      basis_yy_eval[ipt + 5*npts] = x124*(x196*x343 + x404*x90 + x405*y);
      basis_yy_eval[ipt + 6*npts] = 0.0625*x409;
      basis_yy_eval[ipt + 7*npts] = x61*(x194*x410 + x403*x90 + x405);
      basis_yy_eval[ipt + 8*npts] = x413*x62;
      basis_yy_eval[ipt + 9*npts] = x206*(x102*x415 + x209*x416 + x417);
      basis_yy_eval[ipt + 10*npts] = x419*x72;
      basis_yy_eval[ipt + 11*npts] = x74*(x213*x410 + x421*x90 + x422);
      basis_yy_eval[ipt + 12*npts] = x426*x77;

      // Evaluate second derivative of bfn wrt yz
      basis_yz_eval[ipt + 0*npts] = x60*x79*(radial_eval_alpha*x161 + radial_eval_alpha_squared*x153);
      basis_yz_eval[ipt + 1*npts] = x16*(radial_eval_alpha_squared*x21*x25*x6 + x164*x372 + x165);
      basis_yz_eval[ipt + 2*npts] = x22*x60*(radial_eval_alpha*x170 + radial_eval_alpha_squared*x167 + x255*x90 + x318*x427);
      basis_yz_eval[ipt + 3*npts] = x100*(radial_eval*x428 + radial_eval_alpha_squared*x25*x38*x6 + x180*x372 + x228*x315);
      basis_yz_eval[ipt + 4*npts] = x181*(x182*x363 + x188*x236 + x230*x431 + x361*x432);
      basis_yz_eval[ipt + 5*npts] = x44*(radial_eval*x433 + radial_eval_alpha_squared*x25*x50*x6 + x196*x372 + x243*x315);
      basis_yz_eval[ipt + 6*npts] = 0.0625*x201*x370 + 0.0625*x249*x425 + 0.0625*x368*x434 + 0.0625*x435*x59;
      basis_yz_eval[ipt + 7*npts] = x250*(x202*x243 + x238*x340 + x318*x439 + x438*x90);
      basis_yz_eval[ipt + 8*npts] = x62*(x205*x377 + x251*x441 + x374*x440 + x435*x67);
      basis_yz_eval[ipt + 9*npts] = x181*(x102*x443 + x202*x257 + x252*x340 + x323*x444);
      basis_yz_eval[ipt + 10*npts] = x258*(x212*x379 + x340*x73 + x446);
      basis_yz_eval[ipt + 11*npts] = x261*(x214 + x262*x340 + x318*x447);
      basis_yz_eval[ipt + 12*npts] = x383*(x215*x384 + x340*x78);

      // Evaluate second derivative of bfn wrt zz
      basis_zz_eval[ipt + 0*npts] = x1*x449*x79;
      basis_zz_eval[ipt + 1*npts] = x18*(x21*x379 + x450);
      basis_zz_eval[ipt + 2*npts] = x23*(x221*x451 + x222 + x452);
      basis_zz_eval[ipt + 3*npts] = x101*(x228*x379 + x453*x455 + x456);
      basis_zz_eval[ipt + 4*npts] = x229*(x230*x458 + x235*x459 + x460);
      basis_zz_eval[ipt + 5*npts] = x237*x465;
      basis_zz_eval[ipt + 6*npts] = 0.0625*x468;
      basis_zz_eval[ipt + 7*npts] = x250*x465;
      basis_zz_eval[ipt + 8*npts] = x470*x62;
      basis_zz_eval[ipt + 9*npts] = x206*(x257*x379 + x453*x472 + x473);
      basis_zz_eval[ipt + 10*npts] = x474*x72;
      basis_zz_eval[ipt + 11*npts] = x74*(x379*x76 + x475);
      basis_zz_eval[ipt + 12*npts] = x476*x77;

      // Evaluate Laplacian of bfn 
      basis_lapl_eval[ipt + 0*npts] = x79*(x1*x264 + x1*x386 + x1*x449 + x161*x479 + x269*x478 + x390*x477 + x479*x89);
      basis_lapl_eval[ipt + 1*npts] = x145*(x21*x343 + x450*y + x483);
      basis_lapl_eval[ipt + 2*npts] = x22*(x1*x221*x90 + x1*x278 + x1*x396 + x1*x452 + x170*x479 + x221*x272*x484 + x276*x478 + x394*x477 + x479*x99);
      basis_lapl_eval[ipt + 3*npts] = x134*(radial_eval*x204*x455 + x106*x487 + x228*x343 + x282*x485 + x284*y + x399 + x456*y);
      basis_lapl_eval[ipt + 4*npts] = x100*(x1*x230*x458 + x1*x288 + x1*x402 + x1*x460 + x113*x479 + x17*x235*x488 + x188*x479 + x287*x478 + x401*x477);
      basis_lapl_eval[ipt + 5*npts] = x44*(radial_eval*x183*x463 + x117*x489 + x17*x290*x90 + x17*x291 + x17*x405 + x17*x464 + x196*x491 + x243*x491 + x404*x490);
      basis_lapl_eval[ipt + 6*npts] = 0.0625*x297 + 0.0625*x409 + 0.0625*x468;
      basis_lapl_eval[ipt + 7*npts] = x44*(radial_eval*x109*x463 + x128*x492 + x194*x489 + x243*x492 + x291*x60 + x298*x490 + x403*x60*x90 + x405*x60 + x464*x60);
      basis_lapl_eval[ipt + 8*npts] = x62*(x302 + x413 + x470);
      basis_lapl_eval[ipt + 9*npts] = x134*(radial_eval*x131*x472 + x*x417 + x*x473 + x209*x487 + x257*x270 + x305 + x415*x493);
      basis_lapl_eval[ipt + 10*npts] = x72*(x309 + x419 + x474);
      basis_lapl_eval[ipt + 11*npts] = x145*(x*x475 + x270*x76 + x495);
      basis_lapl_eval[ipt + 12*npts] = x77*(x313 + x426 + x476);

      // Evaluate Laplacian gradient of bfn (dx)
      basis_lapl_x_eval[ipt + 0*npts] = x79*(x1*x15*x503 + x161*x506 + x172*x263*x89 + x265*(x390 - 30.0*x4*y) + x269*x502 + x316*x343 + x385*x89*y + x390*x504 + x448*x89*y + x485*(x268 + x496) + x497*x499 + x497*x501);
      basis_lapl_x_eval[ipt + 1*npts] = x145*(x1*x509 + x1*x510 + x164*x508 + x190*x385*x92 + x190*x448*x92 + x21*x503*y + x21*x508 + x211*x263*x92 + x226*x507 + x301*x391 + x317*x410 + x410*x92);
      basis_lapl_x_eval[ipt + 2*npts] = x22*(x1*x30*x503 + x170*x506 + x172*x263*x99 + x226*x24*x410 + x276*x502 + x321*x343 + x353*x445 + x353*x512 + x385*x99*y + x394*x504 + x395 + x448*x99*y + x481 + x485*(x275 + x388) + x497*x513 + x497*x514);
      basis_lapl_x_eval[ipt + 3*npts] = x134*(x1*x518 + x1*x519 + x106*x263*x515 + x106*x385*x424 + x106*x424*x448 + x180*x508 + x228*x508 + x282*x516 + x301*x398 + x322*x416 + x356*x416 + x38*x503*y + x455*x517);
      basis_lapl_x_eval[ipt + 4*npts] = x100*(x1*x43*x503 + x113*x172*x263 + x113*x385*y + x113*x448*y + x188*x506 + x202*x360*x522 - x230*x521*y + x265*(x4*x436 + x401) + x287*x502 + x325*x343 + x362*x523 + x4*x441*x458 + x401*x504 + x485*(x285 + x520) + x497*x524 + x497*x525);
      basis_lapl_x_eval[ipt + 5*npts] = x44*(x117*x263*x530 + x117*x385*x528 + x117*x448*x528 + x17*x526 + x196*x532 + x219*x531 + x290*x533 + x327*x529 + x366*x529 + x404*x527*x93 + x535);
      basis_lapl_x_eval[ipt + 6*npts] = 0.0625*x*x543 + 0.0625*x*x544 + 5.625*x123*x263 + 1.875*x123*x385 + 1.875*x123*x448 + 0.0625*x201*x542 + 1.5*x249*x371 + 0.0625*x295*x541 + 0.0625*x296*x467 - 0.0625*x329*x537 - 0.0625*x329*(-18.0*x*x25 + x137 + x538) + 0.0625*x333*x539 + 0.0625*x334*x407 + 0.0625*x369*x540 + 0.0625*x503*x59 + 0.0625*x536*(x104 + x110);
      basis_lapl_x_eval[ipt + 7*npts] = x44*(x128*x263*x358 + x128*x385*z + x128*x448*z + x243*x3*x550 + x298*x548*x93 + 8.0*x314*x463 + x338*x529 + 40.0*x339*x435 - x341*(x248*x4 - x463) + x373*x377 + x4*x403*x547 + x526*x60 + x531*x549 + x534*x549 + x545*(x289 + x520) + x546);
      basis_lapl_x_eval[ipt + 8*npts] = x62*(x*x555 + x*x556 + 6.0*x133*x263 + 2.0*x133*x385 + 2.0*x133*x448 + 16.0*x202*x342 + x205*x554 + x251*x553 + x270*x411 + x300*x351 + x317*x341 + x374*(x136*x25 + x97) + x375*x551 + x376*x469 + x503*x67 + x552*(-x131*x25 + x538 + x91));
      basis_lapl_x_eval[ipt + 9*npts] = x134*(x*x503*x71 + x102*(x225 + x414) + x102*(x303 - x496) + 3.0*x140*x263 + x140*x385 + x140*x448 + x257*x3*x559 + x3*x557 + x3*x558 + x304*x312 + 16.0*x314*x472 + x323*x4*x415 + 24.0*x340*x345 + x344*x416 + x378*x379 + x453*(x254 + 7.0*x4 + x471));
      basis_lapl_x_eval[ipt + 10*npts] = x72*(x*x560 + x*x561 + 6.0*x144*x263 + 2.0*x144*x385 + 2.0*x144*x448 + 80.0*x202*x332 + x212*x554 + x259*x564 + x308*x334 + x317*x562 + x329*(x115 + x91 + x97) + x346*x537 + x382 + 10.0*x418*x93 + x503*x73);
      basis_lapl_x_eval[ipt + 11*npts] = x145*(x*x503*x76 + x102*(x268 + x387) + x146*x559 + 3.0*x148*x263 + x148*x379 + x148*x385 + x148*x448 + x3*x565 + x3*x566 + x310*x312 + x318*x4*x421 + x347*x410 + x348*x511 + x90*(x253 + x274));
      basis_lapl_x_eval[ipt + 12*npts] = x77*(x*x568 + x*x569 + 18.0*x151*x263 + 6.0*x151*x385 + 6.0*x151*x448 + x215*x567 + x259*x541 + x317*x329 + x329*(x137 + x97) + x334*x423 + x349*x539 + x503*x78);
      // Evaluate Laplacian gradient of bfn (dy)
      basis_lapl_y_eval[ipt + 0*npts] = x79*(x*x161*x263 + x*x161*x448 + x1*x15*x573 + x136*x161*x385 - x265*(30.0*x*x6 - x269) + x269*x574 + x270*x316 + x390*x502 + x493*(x388 + 15.0*x6) + x501*x570 + x570*x572 + x576*x89);
      basis_lapl_y_eval[ipt + 1*npts] = x145*(x102*(x267 + x388) + x162*x559 + x164*x263 + x164*x379 + 3.0*x164*x385 + x164*x448 + x21*x573*y + x226*x318*x6 + x272*x317 + x319*x563 + x391*x425 + x5*x510 + x5*x577 + x90*(x225 + x254));
      basis_lapl_y_eval[ipt + 2*npts] = x22*(x*x170*x263 + x*x170*x448 + x1*x30*x573 + x136*x170*x385 + x24*x255*x272 + x255*x494 + x270*x321 + x276*x574 + x277 + x381*x427 + x394*x502 + x427*x564 + x493*(x267 + x393) + x514*x570 + x570*x578 + x576*x99);
      basis_lapl_y_eval[ipt + 3*npts] = x134*(-x102*(x253 + x281) - x102*(x397 - 15.0*x6) + x180*x263 + 3.0*x180*x385 + x180*x448 + x228*x5*x559 + x282*x323*x6 + x283*x322 + 16.0*x315*x455 + 24.0*x320*x324 + x379*x428 + x38*x573*y + x398*x425 + x453*(x225 + x454 - 7.0*x6) + x5*x519 + x5*x579);
      basis_lapl_y_eval[ipt + 4*npts] = x100*(x*x188*x263 + x*x188*x448 - x*x230*x581 + x1*x43*x573 + x113*x576 + x136*x188*x385 + x265*(x287 + x364*x6) + x270*x325 + x287*x574 + x376*x458*x6 + x401*x502 + x431*x522*x93 + x432*x553 + x493*(x285 + x580) + x525*x570 + x570*x582);
      basis_lapl_y_eval[ipt + 5*npts] = x44*(x17*x586 + x196*x263*z + x196*x358*x385 + x196*x448*z + x202*x404*x548 + x243*x5*x550 + x290*x547*x6 + 8.0*x315*x463 + x327*x583 + 40.0*x328*x371 - x341*(x248*x6 - x463) + x377*x433 + x534*x584 + x545*(x289 + x580) + x546 + x584*x585);
      basis_lapl_y_eval[ipt + 6*npts] = 0.0625*x123*x542 + 1.875*x201*x263 + 5.625*x201*x385 + 1.875*x201*x448 + 1.5*x249*x435 + 0.0625*x295*x335 - 0.0625*x329*x587 - 0.0625*x329*(x173 - 18.0*x25*y + x588) + 0.0625*x333*x589 + 0.0625*x407*x590 + 0.0625*x408*x467 + 0.0625*x434*x540 + 0.0625*x536*(x185 + x207) + 0.0625*x544*y + 0.0625*x573*x59 + 0.0625*x591*y;
      basis_lapl_y_eval[ipt + 7*npts] = x44*(x128*x532 + x143*x194*x385*z + x190*x236*x298 + x194*x263*x592 + x194*x448*x592 + x219*x585 + x338*x583 + x403*x533 + x438*x583 + x535 + x586*x60);
      basis_lapl_y_eval[ipt + 8*npts] = x62*(x133*x554 + 2.0*x205*x263 + 6.0*x205*x385 + 2.0*x205*x448 + x251*x523 + x300*x343 + x341*x594 + 16.0*x342*x93 + x350*x411 + x374*(x155 - x172*x25) + x440*x551 + x441*x469 - x552*(x184 - 16.0*x25*y + x588) + x556*y + x573*x67 + x595*y);
      basis_lapl_y_eval[ipt + 9*npts] = x134*(36.0*x*x209*x385 + x*x573*x71 + x1*x558 + x1*x596 + x140*x508 + x209*x263*x354 + x209*x354*x448 + x257*x508 + x283*x344 + x283*x443 + x304*x412 + x415*x516 + x472*x517);
      basis_lapl_y_eval[ipt + 10*npts] = x72*(x144*x554 + 10.0*x202*x308 + 2.0*x212*x263 + 6.0*x212*x385 + 2.0*x212*x448 + x259*x512 + x329*(x184 + x192 + x593) + 80.0*x332*x93 + x335*x418 + x346*x587 + x347*x562 + x446 + x561*y + x573*x73 + x597*y);
      basis_lapl_y_eval[ipt + 11*npts] = x145*(x*x573*x76 + x1*x566 + x1*x598 + x126*x213*x263 + x126*x213*x448 + x143*x213*x385 + x148*x508 + x213*x272 + x272*x347 + x310*x412 + x421*x507 + x508*x76);
      basis_lapl_y_eval[ipt + 12*npts] = x77*(x151*x567 + 6.0*x215*x263 + 18.0*x215*x385 + 6.0*x215*x448 + x259*x335 + x329*x347 + x329*x594 + x349*x589 + x423*x590 + x569*y + x573*x78 + x599*y);
      // Evaluate Laplacian gradient of bfn (dz)
      basis_lapl_z_eval[ipt + 0*npts] = x79*(x1*x15*x602 + x161*x532 + x219*x499 + x219*x572 + x269*x601 + x390*x600 + x532*x89);
      basis_lapl_z_eval[ipt + 1*npts] = x16*(x164*x604 + x17*x21*x602 + x172*x450 + x226*x25*x445 + x352*x605 + x391*x603 + x483 + x484*x509 + x484*x577);
      basis_lapl_z_eval[ipt + 2*npts] = x22*(x1*x30*x602 + x143*x17*x221*x448 + x170*x532 + x219*x513 + x219*x578 + x221*x263*x606 + x221*x385*x606 + x221*x533 + x226*x489 + x255*x489 + x276*x601 + x394*x600 + x532*x99);
      basis_lapl_z_eval[ipt + 3*npts] = x100*(x17*x38*x602 + x172*x228*x448 + x180*x604 + x202*x455*x609 + x226*x607*y + x228*x263*y + x228*x385*y + x25*x282*x425 - x265*(x25*x486 - x398) + x343*x428 + x356*x487 + x357*x608 + x398*x603 + x484*x518 + x484*x579 - x485*(x266 + x461 + x6));
      basis_lapl_z_eval[ipt + 4*npts] = x100*(96.0*x1*x235*x448 + x1*x43*x602 + x113*x532 + 96.0*x17*x458*x93 + x188*x532 + x219*x524 + x219*x582 + x235*x263*x611 + x235*x385*x611 + x287*x601 + x360*x610 + x401*x600 + x431*x610);
      basis_lapl_z_eval[ipt + 5*npts] = x44*(x17*x614 + x172*x617 + x196*x604 + x25*x290*x445 + x343*x433 + x366*x482 + x367*x605 + x404*x613 + x416*x463*z + x480*x521 + x484*x531 + x484*x585 + x552*x612*y + x615*y + x616*y + x90*(x404 + x442));
      basis_lapl_z_eval[ipt + 6*npts] = -60.0*radial_eval*(x359 + x430) + 0.0625*x123*x143*x363 + 0.0625*x201*x211*x363 + 30.0*x202*x434 + 11.25*x236*x467 + 0.75*x249*x263 + 0.75*x249*x385 + 2.25*x249*x448 + 0.0625*x295*x370 + 0.0625*x368*(x239 + x359 + x6*x618) + 0.0625*x368*(x239 + x4*x618 + x429) + 30.0*x369*x93 + 0.0625*x370*x407 + 0.0625*x543*z + 0.0625*x59*x602 + 0.0625*x591*z;
      basis_lapl_z_eval[ipt + 7*npts] = x44*(x*x552*x612 + x*x615 + x*x616 + x128*x620 + x136*x617 + x25*x381*x403 + x270*x373 + x283*x463*z + x298*x613 + x438*x482 + x439*x605 + x494*x581 + x531*x619 + x585*x619 + x60*x614 + x90*(x298 + x355));
      basis_lapl_z_eval[ipt + 8*npts] = x62*(384.0*radial_eval*x221*z + x133*x622 + 256.0*x202*x440 + x205*x621 + 96.0*x236*x469 + 32.0*x251*x263 + 32.0*x251*x385 + 96.0*x251*x448 + x300*x377 + x374*(x232 - x359) - x374*(x232 - x429) + 256.0*x375*x93 + x377*x411 + x555*z + x595*z + x602*x67);
      basis_lapl_z_eval[ipt + 9*npts] = x100*(x*x255*x607 + x*x257*x263 + x*x257*x385 + x136*x257*x448 + x140*x620 + x25*x312*x415 + x265*(24.0*x*x25 + x304) + x270*x378 + x304*x603 + x443*x487 + x444*x608 + x472*x609*x93 + x493*(x389 + x4 + x461) + x557*x619 + x596*x619 + x60*x602*x71);
      basis_lapl_z_eval[ipt + 10*npts] = x72*(x144*x622 + x202*x347*x624 + x212*x621 + x221*x623 + 60.0*x236*x259 + x259*x263*x527 + x259*x385*x527 + x259*x448*x548 + x308*x625 + x317*x624*x93 + x418*x625 + x560*z + x597*z + x602*x73 + x623*(x420 + x6));
      basis_lapl_z_eval[ipt + 11*npts] = x16*(x136*x475 + x148*x620 + x25*x381*x421 + x310*x603 + x447*x605 + x495 + x565*x619 + x598*x619 + x60*x602*x76);
      basis_lapl_z_eval[ipt + 12*npts] = x77*(x151*x354*x363 + x215*x363*x424 + x259*x370 + x370*x423 + x568*z + x599*z + x602*x78);




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
      dang_eval_y_0 = x152*(radial_eval*x161 + radial_eval_alpha*x153);
      dang_eval_z_0 = x15*x216*x93;
      dang_eval_x_1 = x18*x94;
      dang_eval_y_1 = x145*x165;
      dang_eval_z_1 = x217*(radial_eval*x21 + radial_eval_alpha*x218);
      dang_eval_x_2 = x95*(radial_eval*x99 + radial_eval_alpha*x96);
      dang_eval_y_2 = x166*(radial_eval*x170 + radial_eval_alpha*x167);
      dang_eval_z_2 = x219*x22*(radial_eval_alpha*x30 + x222);
      dang_eval_x_3 = x101*(x102*x106 + x38*x93);
      dang_eval_y_3 = x134*(radial_eval*x180 + radial_eval_alpha*x171);
      dang_eval_z_3 = x107*(radial_eval*x228 + radial_eval_alpha*x223);
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

      dang_eval_x_0 = x107*(radial_eval*x113 + radial_eval_alpha*x108);
      dang_eval_y_0 = x181*(radial_eval*x188 + radial_eval_alpha*x182);
      dang_eval_z_0 = x229*(x230*x235 + x236*x43);
      dang_eval_x_1 = x45*(x117*x90 + x50*x93);
      dang_eval_y_1 = x124*(radial_eval*x196 + radial_eval_alpha*x189);
      dang_eval_z_1 = x237*x244;
      dang_eval_x_2 = 0.0625*x118*x123 + 0.0625*x59*x93;
      dang_eval_y_2 = 0.0625*x118*x201 + 0.0625*x202*x59;
      dang_eval_z_2 = 0.0625*x102*x249 + 0.0625*x236*x59;
      dang_eval_x_3 = x124*(radial_eval*x128 + radial_eval_alpha*x125);
      dang_eval_y_3 = x61*(x194*x90 + x202*x50);
      dang_eval_z_3 = x244*x250;
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

      dang_eval_x_0 = x62*(x129*x133 + x67*x93);
      dang_eval_y_0 = x62*(x129*x205 + x202*x67);
      dang_eval_z_0 = x62*(x230*x251 + x236*x67);
      dang_eval_x_1 = x134*(radial_eval*x140 + radial_eval_alpha*x135);
      dang_eval_y_1 = x206*(x102*x209 + x202*x71);
      dang_eval_z_1 = x181*(radial_eval*x257 + radial_eval_alpha*x252);
      dang_eval_x_2 = x72*(x129*x144 + x73*x93);
      dang_eval_y_2 = x72*(x129*x212 + x202*x73);
      dang_eval_z_2 = x258*(radial_eval_alpha*x73 + x260);
      dang_eval_x_3 = x145*x149;
      dang_eval_y_3 = x214*x74;
      dang_eval_z_3 = x261*(radial_eval*x76 + radial_eval_alpha*x262);
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

      dang_eval_x_0 = x77*(x150*x151 + x78*x93);
      dang_eval_y_0 = x77*(x150*x215 + x202*x78);
      dang_eval_z_0 = x236*x77*x78;
      basis_x_eval[ipt + 12*npts] = dang_eval_x_0;
      basis_y_eval[ipt + 12*npts] = dang_eval_y_0;
      basis_z_eval[ipt + 12*npts] = dang_eval_z_0;

#endif
    } // Loop over points within task
  } // Loop over tasks
        
  } // Loop over shells
} // end kernel

} // namespace GauXC
