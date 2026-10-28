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


__global__ __launch_bounds__(128,2) void collocation_device_shell_to_task_kernel_spherical_lapgrad_5(
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
      const auto x13 = x*y; 
      const auto x14 = radial_eval*x13; 
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
      const auto x28 = -2.0*x22; 
      const auto x29 = x28 + x7; 
      const auto x30 = -x29 - x5; 
      const auto x31 = 0.125*sqrt_15; 
      const auto x32 = x31*y; 
      const auto x33 = 8.0*(z*z*z*z); 
      const auto x34 = x22*x5; 
      const auto x35 = x2 + x3; 
      const auto x36 = x20 - 12.0*x23 + x33 - 12.0*x34 + x35; 
      const auto x37 = 0.125*z; 
      const auto x38 = 15.0*x2 - 40.0*x23 + 15.0*x3 + x33 - 40.0*x34 + 30.0*x8; 
      const auto x39 = x*x31; 
      const auto x40 = 0.25*sqrt_105; 
      const auto x41 = x40*z; 
      const auto x42 = -x3; 
      const auto x43 = x2 - 2.0*x23 + 2.0*x34 + x42; 
      const auto x44 = x*x17; 
      const auto x45 = 3.0*x2 + x20 - 24.0*x23 + 8.0*x34 + x42; 
      const auto x46 = 0.375*sqrt_35; 
      const auto x47 = x46*z; 
      const auto x48 = x35 - 6.0*x8; 
      const auto x49 = x*x0; 
      const auto x50 = 5.0*x2 + x3 + x9; 
      const auto x51 = 20.0*radial_eval; 
      const auto x52 = x*x*x; 
      const auto x53 = x*x7; 
      const auto x54 = x52 - x53; 
      const auto x55 = radial_eval_alpha*x; 
      const auto x56 = y*z; 
      const auto x57 = x11*x56; 
      const auto x58 = 3.0*x5; 
      const auto x59 = x15 + x58; 
      const auto x60 = x16*x5; 
      const auto x61 = radial_eval*x59 + radial_eval_alpha*x60; 
      const auto x62 = 4.0*radial_eval; 
      const auto x63 = 3.0*x52; 
      const auto x64 = -12.0*x*x22 + x63; 
      const auto x65 = -x53 - x64; 
      const auto x66 = x26*x56; 
      const auto x67 = -x29 - x58; 
      const auto x68 = -6.0*x*x22 + x52; 
      const auto x69 = x53 + x68; 
      const auto x70 = 3.0*x; 
      const auto x71 = x7*x70; 
      const auto x72 = 4.0*x; 
      const auto x73 = x22*x72; 
      const auto x74 = -x73; 
      const auto x75 = x63 + x71 + x74; 
      const auto x76 = x36*x5; 
      const auto x77 = x36 + x69*x72; 
      const auto x78 = -x52; 
      const auto x79 = x*x22 + x78; 
      const auto x80 = x45*x5; 
      const auto x81 = x53 + x73 + x78; 
      const auto x82 = x45 + x72*x81; 
      const auto x83 = x52 - x71; 
      const auto x84 = x48*x55 + x62*x83; 
      const auto x85 = x5*x50; 
      const auto x86 = -5.0*x*x7 + x52; 
      const auto x87 = x50 + x72*x86; 
      const auto x88 = x10*x7; 
      const auto x89 = 4.0*y; 
      const auto x90 = y*y*y; 
      const auto x91 = -5.0*x5*y + x90; 
      const auto x92 = x10 + x89*x91; 
      const auto x93 = x*z; 
      const auto x94 = x11*x93; 
      const auto x95 = 3.0*x7; 
      const auto x96 = -x95; 
      const auto x97 = x5 + x96; 
      const auto x98 = x16*x7; 
      const auto x99 = radial_eval*x97 + radial_eval_alpha*x98; 
      const auto x100 = x25*x7; 
      const auto x101 = x5*y; 
      const auto x102 = x22*x89; 
      const auto x103 = -x101 - x102 + x90; 
      const auto x104 = x103*x89 - x19 + x2 - x20 - x24 + 24.0*x34; 
      const auto x105 = x26*x93; 
      const auto x106 = x28 + x95; 
      const auto x107 = -x106 - x5; 
      const auto x108 = x36*x7; 
      const auto x109 = -6.0*x22*y + x90; 
      const auto x110 = x101 + x109; 
      const auto x111 = x110*x89 + x36; 
      const auto x112 = -x102; 
      const auto x113 = 3.0*x90; 
      const auto x114 = 3.0*y; 
      const auto x115 = x114*x5; 
      const auto x116 = x113 + x115; 
      const auto x117 = x112 + x116; 
      const auto x118 = radial_eval_alpha*y; 
      const auto x119 = -x22*y + x90; 
      const auto x120 = 12.0*y; 
      const auto x121 = -x120*x22; 
      const auto x122 = x101 + x113 + x121; 
      const auto x123 = -x115 + x90; 
      const auto x124 = x118*x48 + x123*x62; 
      const auto x125 = -x101 + x90; 
      const auto x126 = x0*z; 
      const auto x127 = x11*x13; 
      const auto x128 = x16*x22; 
      const auto x129 = x17*x56; 
      const auto x130 = 16.0*radial_eval; 
      const auto x131 = x130*x59; 
      const auto x132 = x13*x26; 
      const auto x133 = -6.0*x22; 
      const auto x134 = x133 + x7; 
      const auto x135 = -x134 - x5; 
      const auto x136 = 8.0*radial_eval; 
      const auto x137 = 3.0*z; 
      const auto x138 = x137*x5; 
      const auto x139 = z*z*z; 
      const auto x140 = -4.0*x139; 
      const auto x141 = x137*x7 + x140; 
      const auto x142 = -x138 - x141; 
      const auto x143 = radial_eval_alpha*z; 
      const auto x144 = x136*x142 + x143*x36; 
      const auto x145 = x22*x38; 
      const auto x146 = 16.0*z; 
      const auto x147 = 5.0*z; 
      const auto x148 = -x146*(-2.0*x139 + x147*x5 + x147*x7) + x38; 
      const auto x149 = x22*x43; 
      const auto x150 = 4.0*x16*x21 + x43; 
      const auto x151 = x17*x93; 
      const auto x152 = x130*x97; 
      const auto x153 = x22*x48; 
      const auto x154 = 40.0*x55; 
      const auto x155 = radial_eval_alpha + radial_eval_alpha_squared*x5; 
      const auto x156 = x10*x155; 
      const auto x157 = x12*x13; 
      const auto x158 = 6.0*radial_eval; 
      const auto x159 = 2.0*radial_eval_alpha; 
      const auto x160 = x159*x59; 
      const auto x161 = x155*x16 + x160; 
      const auto x162 = -12.0*x22; 
      const auto x163 = 9.0*x5; 
      const auto x164 = x162 + x163; 
      const auto x165 = -x164 - x7; 
      const auto x166 = 8.0*x55; 
      const auto x167 = x155*x25; 
      const auto x168 = x13*x27; 
      const auto x169 = -x158; 
      const auto x170 = x155*x30 + x159*x67; 
      const auto x171 = x134 + x58; 
      const auto x172 = x155*x36; 
      const auto x173 = 4.0*x22; 
      const auto x174 = -x173; 
      const auto x175 = x174 + x95; 
      const auto x176 = x163 + x175; 
      const auto x177 = x155*x38; 
      const auto x178 = x*x171 + 2.0*x69; 
      const auto x179 = 2.0*x55; 
      const auto x180 = x*x172 + x178*x62 + x179*x77; 
      const auto x181 = -x58; 
      const auto x182 = x181 + x22; 
      const auto x183 = x155*x43 + x166*x79 + x182*x62; 
      const auto x184 = x173 + x7; 
      const auto x185 = x*(x181 + x184) + 2.0*x81; 
      const auto x186 = x*x155*x45 + x179*x82 + x185*x62; 
      const auto x187 = 12.0*radial_eval; 
      const auto x188 = x16*x187; 
      const auto x189 = x155*x48 + x166*x83 + x188; 
      const auto x190 = 5.0*x7; 
      const auto x191 = -x190; 
      const auto x192 = x*(x191 + x58) + 2.0*x86; 
      const auto x193 = x*x155*x50 + x179*x87 + x192*x62; 
      const auto x194 = 20.0*radial_eval_alpha; 
      const auto x195 = x54*x7; 
      const auto x196 = radial_eval_alpha_squared*x; 
      const auto x197 = x58 + x96; 
      const auto x198 = radial_eval_alpha*x4; 
      const auto x199 = radial_eval_alpha*x6; 
      const auto x200 = -x64 - x71; 
      const auto x201 = 4.0*radial_eval_alpha; 
      const auto x202 = x65*x7; 
      const auto x203 = -x106 - x58; 
      const auto x204 = x68 + x71; 
      const auto x205 = x69*x7; 
      const auto x206 = x196*y; 
      const auto x207 = x109 + x115; 
      const auto x208 = x110*x5; 
      const auto x209 = radial_eval_alpha_squared*y; 
      const auto x210 = 4.0*x55; 
      const auto x211 = radial_eval_alpha*x89; 
      const auto x212 = x116 + x121; 
      const auto x213 = x122*x5; 
      const auto x214 = 24.0*radial_eval; 
      const auto x215 = x125*x5; 
      const auto x216 = radial_eval_alpha*x21; 
      const auto x217 = 96.0*radial_eval; 
      const auto x218 = 16.0*x55; 
      const auto x219 = -x171; 
      const auto x220 = 48.0*x93; 
      const auto x221 = 4.0*x143; 
      const auto x222 = x196*z; 
      const auto x223 = 60.0*radial_eval; 
      const auto x224 = x52 + x53 + x74; 
      const auto x225 = x22*x75; 
      const auto x226 = 9.0*z; 
      const auto x227 = -x141 - x226*x5; 
      const auto x228 = 8.0*radial_eval_alpha; 
      const auto x229 = x142*x5; 
      const auto x230 = radial_eval_alpha_squared*z; 
      const auto x231 = x22*x70 + x78; 
      const auto x232 = x22*x79; 
      const auto x233 = x17*z; 
      const auto x234 = x130*x197; 
      const auto x235 = 16.0*radial_eval_alpha; 
      const auto x236 = x5*x97; 
      const auto x237 = x234 + x235*x236; 
      const auto x238 = x22*x83; 
      const auto x239 = radial_eval_alpha + radial_eval_alpha_squared*x7; 
      const auto x240 = 5.0*x5; 
      const auto x241 = -x240; 
      const auto x242 = 2.0*x91 + y*(x241 + x95); 
      const auto x243 = 2.0*x118; 
      const auto x244 = x10*x239*y + x242*x62 + x243*x92; 
      const auto x245 = x159*x97; 
      const auto x246 = x16*x239 + x245; 
      const auto x247 = 2.0*x103 - y*(x173 + x97); 
      const auto x248 = x104*x243 + x239*x25*y + x247*x62; 
      const auto x249 = x107*x159 + x239*x30; 
      const auto x250 = x133 + x95; 
      const auto x251 = x250 + x5; 
      const auto x252 = 2.0*x110 + x251*y; 
      const auto x253 = x239*x36; 
      const auto x254 = x111*x243 + x252*x62 + x253*y; 
      const auto x255 = 9.0*x7; 
      const auto x256 = x174 + x58; 
      const auto x257 = x255 + x256; 
      const auto x258 = 40.0*x118; 
      const auto x259 = x239*x38; 
      const auto x260 = 8.0*x118; 
      const auto x261 = -x22 + x95; 
      const auto x262 = x119*x260 + x239*x43 + x261*x62; 
      const auto x263 = x162 + x255; 
      const auto x264 = x263 + x5; 
      const auto x265 = x239*x45; 
      const auto x266 = -x5; 
      const auto x267 = x266 + x7; 
      const auto x268 = x123*x260 + x187*x267 + x239*x48; 
      const auto x269 = x266 + x95; 
      const auto x270 = x239*x50; 
      const auto x271 = x59*x7; 
      const auto x272 = x234 + x235*x271; 
      const auto x273 = -x251; 
      const auto x274 = -x138 - x140 - x226*x7; 
      const auto x275 = x142*x7; 
      const auto x276 = x101 + x112 + x90; 
      const auto x277 = x117*x22; 
      const auto x278 = 48.0*x56; 
      const auto x279 = x209*z; 
      const auto x280 = -x114*x22 + x90; 
      const auto x281 = x119*x22; 
      const auto x282 = 16.0*x118; 
      const auto x283 = x123*x22; 
      const auto x284 = radial_eval_alpha + radial_eval_alpha_squared*x22; 
      const auto x285 = x10*x284; 
      const auto x286 = x16*x284; 
      const auto x287 = x159*x16 + x286; 
      const auto x288 = 32.0*x216; 
      const auto x289 = x25*x284; 
      const auto x290 = x135*x159 + x284*x30; 
      const auto x291 = -x174 - x5 - x7; 
      const auto x292 = 16.0*x143; 
      const auto x293 = x284*x36; 
      const auto x294 = x142*x292 + x214*x291 + x293; 
      const auto x295 = 10.0*z; 
      const auto x296 = x133 + x190; 
      const auto x297 = 4.0*x139 - x295*x5 - x295*x7 - z*(x240 + x296); 
      const auto x298 = x130*x297 + 2.0*x143*x148 + x284*x38*z; 
      const auto x299 = 3.0*x16; 
      const auto x300 = x150*x159 + x284*x43 + x299*x62; 
      const auto x301 = x284*x45; 
      const auto x302 = x284*x48; 
      const auto x303 = x159*x48 + x302; 
      const auto x304 = x284*x50; 
      const auto x305 = x154*y; 
      const auto x306 = x161 + x246; 
      const auto x307 = x62*y; 
      const auto x308 = 8.0*y; 
      const auto x309 = x308*x55; 
      const auto x310 = 32.0*x21; 
      const auto x311 = x51*z; 
      const auto x312 = x*x62; 
      const auto x313 = x189 + x268; 
      const auto x314 = 60.0*y; 
      const auto x315 = x314*x55; 
      const auto x316 = 20.0*y; 
      const auto x317 = radial_eval_alpha_cubed*x7 + radial_eval_alpha_squared; 
      const auto x318 = x10*x317; 
      const auto x319 = radial_eval_alpha_cubed*x22 + radial_eval_alpha_squared; 
      const auto x320 = x10*x319; 
      const auto x321 = radial_eval_alpha_cubed*x52 + radial_eval_alpha_squared*x70; 
      const auto x322 = 2.0*x206; 
      const auto x323 = 12.0*radial_eval_alpha; 
      const auto x324 = x323*x5; 
      const auto x325 = x159*x197; 
      const auto x326 = 2.0*radial_eval_alpha_squared; 
      const auto x327 = x326*x4; 
      const auto x328 = x239*x59; 
      const auto x329 = x284*x59; 
      const auto x330 = x155*x59; 
      const auto x331 = x16*x317; 
      const auto x332 = x16*x319; 
      const auto x333 = x55*y; 
      const auto x334 = 192.0*x22*x333; 
      const auto x335 = x218*y; 
      const auto x336 = x206*x310; 
      const auto x337 = x120*x55; 
      const auto x338 = x25*x317; 
      const auto x339 = x25*x319; 
      const auto x340 = x159*x203; 
      const auto x341 = x159*x219; 
      const auto x342 = x30*x317; 
      const auto x343 = x30*x319; 
      const auto x344 = x321*x36; 
      const auto x345 = x317*x36; 
      const auto x346 = x319*x36; 
      const auto x347 = 24.0*y; 
      const auto x348 = x13*x346 + x142*x146*x206 - 96.0*x22*x333 + x291*x347*x55; 
      const auto x349 = 240.0*z; 
      const auto x350 = 120.0*x143; 
      const auto x351 = 20.0*z; 
      const auto x352 = x351*x55; 
      const auto x353 = 60.0*z; 
      const auto x354 = 40.0*x196; 
      const auto x355 = x354*x56; 
      const auto x356 = x317*x38; 
      const auto x357 = x319*x38; 
      const auto x358 = 2.0*x222; 
      const auto x359 = x256 + x7; 
      const auto x360 = x62*(x250 + x58); 
      const auto x361 = 24.0*radial_eval_alpha; 
      const auto x362 = 8.0*x209; 
      const auto x363 = radial_eval_alpha_squared*x146; 
      const auto x364 = 12.0*x55; 
      const auto x365 = x196*x308; 
      const auto x366 = x317*x43; 
      const auto x367 = x319*x43; 
      const auto x368 = 2.0*x196; 
      const auto x369 = x317*x45; 
      const auto x370 = x319*x45; 
      const auto x371 = x197*x288; 
      const auto x372 = radial_eval_alpha_squared*x310; 
      const auto x373 = 48.0*x55; 
      const auto x374 = x317*x48; 
      const auto x375 = x319*x48; 
      const auto x376 = 40.0*x209; 
      const auto x377 = x317*x50; 
      const auto x378 = x319*x50; 
      const auto x379 = radial_eval_alpha_cubed*x5 + radial_eval_alpha_squared; 
      const auto x380 = x10*x379; 
      const auto x381 = 12.0*x118; 
      const auto x382 = radial_eval_alpha_cubed*x90 + radial_eval_alpha_squared*x114; 
      const auto x383 = -x323*x7; 
      const auto x384 = x326*x6; 
      const auto x385 = x155*x97; 
      const auto x386 = x284*x97; 
      const auto x387 = x239*x97; 
      const auto x388 = x16*x379; 
      const auto x389 = x25*x379; 
      const auto x390 = 8.0*x196; 
      const auto x391 = x159*x273; 
      const auto x392 = x30*x379; 
      const auto x393 = x175 + x5; 
      const auto x394 = x36*x379; 
      const auto x395 = x36*x382; 
      const auto x396 = x118*x351; 
      const auto x397 = x379*x38; 
      const auto x398 = 2.0*x279; 
      const auto x399 = 12.0*x; 
      const auto x400 = x379*x43; 
      const auto x401 = 2.0*x209; 
      const auto x402 = x379*x45; 
      const auto x403 = 48.0*x118; 
      const auto x404 = x379*x48; 
      const auto x405 = 20.0*x; 
      const auto x406 = x379*x50; 
      const auto x407 = radial_eval_alpha_cubed*x139 + radial_eval_alpha_squared*x137; 
      const auto x408 = x21*x326; 
      const auto x409 = x118*z; 
      const auto x410 = x143*x89; 
      const auto x411 = x146*y; 
      const auto x412 = x390*x56; 
      const auto x413 = x142*x155; 
      const auto x414 = x142*x239; 
      const auto x415 = x142*x284; 
      const auto x416 = x36*x407; 
      const auto x417 = x55*z; 
      const auto x418 = x210*z; 
      const auto x419 = 8.0*x; 
      const auto x420 = 3.0*x22; 
      const auto x421 = x*x146; 


      // Evaluate basis function
      basis_eval[ipt + 0*npts] = radial_eval*x1*x10;
      basis_eval[ipt + 1*npts] = x12*x14*x16;
      basis_eval[ipt + 2*npts] = radial_eval*x18*x25;
      basis_eval[ipt + 3*npts] = x14*x27*x30;
      basis_eval[ipt + 4*npts] = radial_eval*x32*x36;
      basis_eval[ipt + 5*npts] = radial_eval*x37*x38;
      basis_eval[ipt + 6*npts] = radial_eval*x36*x39;
      basis_eval[ipt + 7*npts] = radial_eval*x41*x43;
      basis_eval[ipt + 8*npts] = radial_eval*x44*x45;
      basis_eval[ipt + 9*npts] = radial_eval*x47*x48;
      basis_eval[ipt + 10*npts] = radial_eval*x49*x50;


    
      // Evaluate first derivative of bfn wrt x
      basis_x_eval[ipt + 0*npts] = x1*(x10*x55 + x51*x54);
      basis_x_eval[ipt + 1*npts] = x57*x61;
      basis_x_eval[ipt + 2*npts] = x18*(x25*x55 + x62*x65);
      basis_x_eval[ipt + 3*npts] = x66*(radial_eval*x67 + radial_eval_alpha*x30*x5);
      basis_x_eval[ipt + 4*npts] = x32*(x36*x55 + x62*x69);
      basis_x_eval[ipt + 5*npts] = x37*(x38*x55 + x51*x75);
      basis_x_eval[ipt + 6*npts] = x31*(radial_eval*x77 + radial_eval_alpha*x76);
      basis_x_eval[ipt + 7*npts] = x41*(x43*x55 + x62*x79);
      basis_x_eval[ipt + 8*npts] = x17*(radial_eval*x82 + radial_eval_alpha*x80);
      basis_x_eval[ipt + 9*npts] = x47*x84;
      basis_x_eval[ipt + 10*npts] = x0*(radial_eval*x87 + radial_eval_alpha*x85);

      // Evaluate first derivative of bfn wrt y
      basis_y_eval[ipt + 0*npts] = x0*(radial_eval*x92 + radial_eval_alpha*x88);
      basis_y_eval[ipt + 1*npts] = x94*x99;
      basis_y_eval[ipt + 2*npts] = x17*(radial_eval*x104 + radial_eval_alpha*x100);
      basis_y_eval[ipt + 3*npts] = x105*(radial_eval*x107 + radial_eval_alpha*x30*x7);
      basis_y_eval[ipt + 4*npts] = x31*(radial_eval*x111 + radial_eval_alpha*x108);
      basis_y_eval[ipt + 5*npts] = x37*(x117*x51 + x118*x38);
      basis_y_eval[ipt + 6*npts] = x39*(x110*x62 + x118*x36);
      basis_y_eval[ipt + 7*npts] = x41*(x118*x43 + x119*x62);
      basis_y_eval[ipt + 8*npts] = x44*(x118*x45 + x122*x62);
      basis_y_eval[ipt + 9*npts] = x124*x47;
      basis_y_eval[ipt + 10*npts] = x49*(x118*x50 + x125*x51);

      // Evaluate first derivative of bfn wrt z
      basis_z_eval[ipt + 0*npts] = x10*x118*x126;
      basis_z_eval[ipt + 1*npts] = x127*(radial_eval*x16 + radial_eval_alpha*x128);
      basis_z_eval[ipt + 2*npts] = x129*(radial_eval_alpha*x25 + x131);
      basis_z_eval[ipt + 3*npts] = x132*(radial_eval*x135 + radial_eval_alpha*x22*x30);
      basis_z_eval[ipt + 4*npts] = x144*x32;
      basis_z_eval[ipt + 5*npts] = 0.125*radial_eval*x148 + 0.125*radial_eval_alpha*x145;
      basis_z_eval[ipt + 6*npts] = x144*x39;
      basis_z_eval[ipt + 7*npts] = x40*(radial_eval*x150 + radial_eval_alpha*x149);
      basis_z_eval[ipt + 8*npts] = x151*(radial_eval_alpha*x45 + x152);
      basis_z_eval[ipt + 9*npts] = x46*(radial_eval*x48 + radial_eval_alpha*x153);
      basis_z_eval[ipt + 10*npts] = x126*x50*x55;

      // Evaluate second derivative of bfn wrt xx
      basis_xx_eval[ipt + 0*npts] = x1*(x154*x54 + x156 + x51*x59);
      basis_xx_eval[ipt + 1*npts] = x157*(x158 + x161);
      basis_xx_eval[ipt + 2*npts] = x18*(x165*x62 + x166*x65 + x167);
      basis_xx_eval[ipt + 3*npts] = x168*(x169 + x170);
      basis_xx_eval[ipt + 4*npts] = x32*(x166*x69 + x171*x62 + x172);
      basis_xx_eval[ipt + 5*npts] = x37*(x154*x75 + x176*x51 + x177);
      basis_xx_eval[ipt + 6*npts] = x180*x31;
      basis_xx_eval[ipt + 7*npts] = x183*x41;
      basis_xx_eval[ipt + 8*npts] = x17*x186;
      basis_xx_eval[ipt + 9*npts] = x189*x47;
      basis_xx_eval[ipt + 10*npts] = x0*x193;

      // Evaluate second derivative of bfn wrt xy
      basis_xy_eval[ipt + 0*npts] = x0*(x194*x195 + x196*x88 + x51*x83 + x55*x92);
      basis_xy_eval[ipt + 1*npts] = x12*(radial_eval*x197 + radial_eval_alpha_squared*x16*x5*x7 + x198*x97 + x199*x59);
      basis_xy_eval[ipt + 2*npts] = x17*(x100*x196 + x104*x55 + x200*x62 + x201*x202);
      basis_xy_eval[ipt + 3*npts] = x27*(radial_eval*x203 + radial_eval_alpha_squared*x30*x5*x7 + x107*x198 + x199*x67);
      basis_xy_eval[ipt + 4*npts] = x31*(x108*x196 + x111*x55 + x201*x205 + x204*x62);
      basis_xy_eval[ipt + 5*npts] = x37*(20.0*x117*x55 + 20.0*x118*x75 + 120.0*x14 + x206*x38);
      basis_xy_eval[ipt + 6*npts] = x31*(x118*x77 + x201*x208 + x207*x62 + x209*x76);
      basis_xy_eval[ipt + 7*npts] = x41*(x119*x210 + x206*x43 + x211*x79);
      basis_xy_eval[ipt + 8*npts] = x17*(x118*x82 + x201*x213 + x209*x80 + x212*x62);
      basis_xy_eval[ipt + 9*npts] = x47*(x123*x210 - x13*x214 + x206*x48 + x211*x83);
      basis_xy_eval[ipt + 10*npts] = x0*(x118*x87 + x123*x51 + x194*x215 + x209*x85);

      // Evaluate second derivative of bfn wrt xz
      basis_xz_eval[ipt + 0*npts] = x0*x56*(x10*x196 + x194*x54);
      basis_xz_eval[ipt + 1*npts] = x11*y*(radial_eval_alpha_squared*x16*x22*x5 + x216*x59 + x61);
      basis_xz_eval[ipt + 2*npts] = x129*(x*x217 + x196*x25 + x201*x65 + x218*x59);
      basis_xz_eval[ipt + 3*npts] = x26*y*(radial_eval*x219 + radial_eval_alpha_squared*x22*x30*x5 + x135*x198 + x216*x67);
      basis_xz_eval[ipt + 4*npts] = x32*(-radial_eval*x220 + x142*x166 + x221*x69 + x222*x36);
      basis_xz_eval[ipt + 5*npts] = 0.125*x145*x196 + 0.125*x148*x55 + 0.125*x194*x225 + 0.125*x223*x224;
      basis_xz_eval[ipt + 6*npts] = x31*(x136*x227 + x143*x77 + x228*x229 + x230*x76);
      basis_xz_eval[ipt + 7*npts] = x40*(x149*x196 + x150*x55 + x201*x232 + x231*x62);
      basis_xz_eval[ipt + 8*npts] = x233*(radial_eval_alpha*x82 + radial_eval_alpha_squared*x80 + x237);
      basis_xz_eval[ipt + 9*npts] = x46*(x153*x196 + x201*x238 + x84);
      basis_xz_eval[ipt + 10*npts] = x126*(radial_eval_alpha*x87 + radial_eval_alpha_squared*x85);

      // Evaluate second derivative of bfn wrt yy
      basis_yy_eval[ipt + 0*npts] = x0*x244;
      basis_yy_eval[ipt + 1*npts] = x157*(x169 + x246);
      basis_yy_eval[ipt + 2*npts] = x17*x248;
      basis_yy_eval[ipt + 3*npts] = x168*(x169 + x249);
      basis_yy_eval[ipt + 4*npts] = x254*x31;
      basis_yy_eval[ipt + 5*npts] = x37*(x117*x258 + x257*x51 + x259);
      basis_yy_eval[ipt + 6*npts] = x39*(x110*x260 + x251*x62 + x253);
      basis_yy_eval[ipt + 7*npts] = x262*x41;
      basis_yy_eval[ipt + 8*npts] = x44*(x122*x260 + x264*x62 + x265);
      basis_yy_eval[ipt + 9*npts] = x268*x47;
      basis_yy_eval[ipt + 10*npts] = x49*(x125*x258 + x269*x51 + x270);

      // Evaluate second derivative of bfn wrt yz
      basis_yz_eval[ipt + 0*npts] = x126*(radial_eval_alpha*x92 + radial_eval_alpha_squared*x88);
      basis_yz_eval[ipt + 1*npts] = x*x11*(radial_eval_alpha_squared*x16*x22*x7 + x216*x97 + x99);
      basis_yz_eval[ipt + 2*npts] = x233*(radial_eval_alpha*x104 + radial_eval_alpha_squared*x100 + x272);
      basis_yz_eval[ipt + 3*npts] = x*x26*(radial_eval*x273 + radial_eval_alpha_squared*x22*x30*x7 + x107*x216 + x135*x199);
      basis_yz_eval[ipt + 4*npts] = x31*(x108*x230 + x111*x143 + x136*x274 + x228*x275);
      basis_yz_eval[ipt + 5*npts] = 0.125*x118*x148 + 0.125*x145*x209 + 0.125*x194*x277 + 0.125*x223*x276;
      basis_yz_eval[ipt + 6*npts] = x39*(-radial_eval*x278 + x110*x221 + x142*x260 + x279*x36);
      basis_yz_eval[ipt + 7*npts] = x40*(x118*x150 + x149*x209 + x201*x281 + x280*x62);
      basis_yz_eval[ipt + 8*npts] = x151*(x122*x201 + x209*x45 - x217*y + x282*x97);
      basis_yz_eval[ipt + 9*npts] = x46*(x124 + x153*x209 + x201*x283);
      basis_yz_eval[ipt + 10*npts] = x0*x93*(x125*x194 + x209*x50);

      // Evaluate second derivative of bfn wrt zz
      basis_zz_eval[ipt + 0*npts] = x1*x285;
      basis_zz_eval[ipt + 1*npts] = x157*x287;
      basis_zz_eval[ipt + 2*npts] = x18*(x131 + x288*x59 + x289);
      basis_zz_eval[ipt + 3*npts] = x168*(x187 + x290);
      basis_zz_eval[ipt + 4*npts] = x294*x32;
      basis_zz_eval[ipt + 5*npts] = 0.125*x298;
      basis_zz_eval[ipt + 6*npts] = x294*x39;
      basis_zz_eval[ipt + 7*npts] = x300*x41;
      basis_zz_eval[ipt + 8*npts] = x44*(x152 + x288*x97 + x301);
      basis_zz_eval[ipt + 9*npts] = x303*x47;
      basis_zz_eval[ipt + 10*npts] = x304*x49;

      // Evaluate Laplacian of bfn 
      basis_lapl_eval[ipt + 0*npts] = x0*(x156*y + x244 + x285*y + x305*x54 + x51*x59*y);
      basis_lapl_eval[ipt + 1*npts] = x157*(x287 + x306);
      basis_lapl_eval[ipt + 2*npts] = x17*(x118*x310*x59 + x130*x59*y + x165*x307 + x167*y + x248 + x289*y + x309*x65);
      basis_lapl_eval[ipt + 3*npts] = x168*(x170 + x249 + x290);
      basis_lapl_eval[ipt + 4*npts] = x31*(x118*x142*x146 + x171*x307 + x172*y + x214*x291*y + x254 + x293*y + x309*x69);
      basis_lapl_eval[ipt + 5*npts] = 0.125*x117*x258*z + 0.125*x154*x75*z + 0.125*x176*x311 + 0.125*x177*z + 0.125*x257*x311 + 0.125*x259*z + 0.125*x298;
      basis_lapl_eval[ipt + 6*npts] = x31*(x*x214*x291 + x*x253 + x*x293 + x110*x309 + x142*x146*x55 + x180 + x251*x312);
      basis_lapl_eval[ipt + 7*npts] = x41*(x183 + x262 + x300);
      basis_lapl_eval[ipt + 8*npts] = x17*(x*x130*x97 + x*x265 + x*x301 + x122*x309 + x186 + x264*x312 + x310*x55*x97);
      basis_lapl_eval[ipt + 9*npts] = x47*(x303 + x313);
      basis_lapl_eval[ipt + 10*npts] = x0*(x*x269*x51 + x*x270 + x*x304 + x125*x305 + x193);

      // Evaluate Laplacian gradient of bfn (dx)
      basis_lapl_x_eval[ipt + 0*npts] = x0*(x10*x321*y + x13*x318 + x13*x320 + x155*x314*x54 + x210*x242 + x239*x316*x54 + x258*x83 + x284*x316*x54 + x315*x59 + x322*x92);
      basis_lapl_x_eval[ipt + 1*npts] = x57*(x*x16*x321 + x160 + x324 + x325 + x326*x60 + x327*x97 + x328 + x329 + 3.0*x330 + x331*x4 + x332*x4);
      basis_lapl_x_eval[ipt + 2*npts] = x17*(x104*x322 + x120*x155*x65 + x13*x338 + x13*x339 + x165*x337 + x200*x260 + x210*x247 + x239*x65*x89 + x25*x321*y + x284*x65*x89 + x334 + x335*x59 + x336*x59);
      basis_lapl_x_eval[ipt + 3*npts] = x66*(x*x30*x321 + x107*x327 + x135*x327 + 3.0*x155*x67 + x239*x67 + x284*x67 - x324 + x340 + x341 + x342*x4 + x343*x4);
      basis_lapl_x_eval[ipt + 4*npts] = x31*(x111*x322 + x120*x155*x69 + x13*x345 + x171*x337 + x204*x260 + x210*x252 + x239*x69*x89 + x284*x69*x89 + x344*y + x348);
      basis_lapl_x_eval[ipt + 5*npts] = 0.125*x117*x355 + 0.125*x148*x358 + 0.125*x155*x353*x75 + 0.125*x176*x353*x55 + 0.125*x218*x297 + 0.125*x224*x350 + 0.125*x239*x351*x75 + 0.125*x257*x352 + 0.125*x284*x351*x75 + 0.125*x321*x38*z + 0.125*x349*x55*x7 + 0.125*x356*x93 + 0.125*x357*x93;
      basis_lapl_x_eval[ipt + 6*npts] = x31*(x*x344 + 3.0*x155*x77 + x178*x364 + x187*(x134 + x240) + x201*x251*x5 + x207*x260 + x208*x362 - x214*x359 + x227*x292 + x229*x363 + x239*x77 + x284*x77 + x291*x361*x5 + x345*x4 + x346*x4 + x360);
      basis_lapl_x_eval[ipt + 7*npts] = x41*(x*x366 + x*x367 + x119*x365 + x150*x368 + 12.0*x155*x79 + x182*x364 + x210*x261 + x210*x299 + x228*x231 + 4.0*x239*x79 + 4.0*x284*x79 + x321*x43);
      basis_lapl_x_eval[ipt + 8*npts] = x17*(x*x321*x45 + 3.0*x155*x82 + x185*x364 + x187*(x184 + x241) + x201*x264*x5 + x212*x260 + x213*x362 + x236*x372 + x237 + x239*x82 + x284*x82 + x369*x4 + x370*x4 + x371 + x62*(x263 + x58));
      basis_lapl_x_eval[ipt + 9*npts] = x47*(x*x374 + x*x375 + x123*x365 + 12.0*x155*x83 + 36.0*x16*x55 + x228*x83 + 4.0*x239*x83 + x267*x364 + 4.0*x284*x83 + x321*x48 + x368*x48 - x373*x7);
      basis_lapl_x_eval[ipt + 10*npts] = x0*(x*x321*x50 + x123*x258 + 3.0*x155*x87 + x187*(x191 + x240) + x192*x364 + x194*x269*x5 + x215*x376 + x239*x87 + x284*x87 + x377*x4 + x378*x4 + x51*(x181 + x95));
      // Evaluate Laplacian gradient of bfn (dy)
      basis_lapl_y_eval[ipt + 0*npts] = x0*(x10*x382*y + x154*x83 + x155*x92 + x187*(x190 + x241) + x194*x271 + x195*x354 + x197*x51 + 3.0*x239*x92 + x242*x381 + x284*x92 + x320*x6 + x380*x6);
      basis_lapl_y_eval[ipt + 1*npts] = x94*(x16*x382*y + x245 + x325 + x326*x98 + x332*x6 + x383 + x384*x59 + x385 + x386 + 3.0*x387 + x388*x6);
      basis_lapl_y_eval[ipt + 2*npts] = x17*(x104*x155 + 3.0*x104*x239 + x104*x284 + x165*x201*x7 + x166*x200 - x187*(x173 + x191 + x5) + x202*x390 + x247*x381 + x25*x382*y + x271*x372 + x272 + x339*x6 + x371 + x389*x6 - x62*(x164 + x95));
      basis_lapl_y_eval[ipt + 3*npts] = x105*(x107*x155 + 3.0*x107*x239 + x107*x284 + x135*x384 + x30*x382*y + x340 + x343*x6 + x383 + x384*x67 + x391 + x392*x6);
      basis_lapl_y_eval[ipt + 4*npts] = x31*(x111*x155 + 3.0*x111*x239 + x111*x284 + x166*x204 + x171*x201*x7 + x187*(x296 + x5) + x205*x390 - x214*x393 + x252*x381 + x274*x292 + x275*x363 + x291*x361*x7 + x346*x6 + x360 + x394*x6 + x395*y);
      basis_lapl_y_eval[ipt + 5*npts] = 0.125*x117*x155*x351 + 0.125*x117*x239*x353 + 0.125*x117*x284*x351 + 0.125*x118*x257*x353 + 0.125*x118*x349*x5 + 0.125*x148*x398 + 0.125*x176*x396 + 0.125*x276*x350 + 0.125*x282*x297 + 0.125*x355*x75 + 0.125*x357*x56 + 0.125*x38*x382*z + 0.125*x397*x56;
      basis_lapl_y_eval[ipt + 6*npts] = x31*(x*x395 + x110*x155*x72 + x110*x239*x399 + x110*x284*x72 + x13*x394 + x166*x207 + x178*x211 + x251*x337 + x322*x77 + x348);
      basis_lapl_y_eval[ipt + 7*npts] = x41*(4.0*x119*x155 + 12.0*x119*x239 + 4.0*x119*x284 + x150*x401 + x182*x211 + x211*x299 + x228*x280 + x261*x381 + x365*x79 + x367*y + x382*x43 + x400*y);
      basis_lapl_y_eval[ipt + 8*npts] = x17*(x*x382*x45 + x122*x155*x72 + x122*x239*x399 + x122*x284*x72 + x13*x370 + x13*x402 + x166*x212 + x185*x211 + x264*x337 + x322*x82 - x334 + x335*x97 + x336*x97);
      basis_lapl_y_eval[ipt + 9*npts] = x47*(36.0*x118*x267 + 4.0*x123*x155 + x123*x228 + 12.0*x123*x239 + 4.0*x123*x284 + x16*x381 + x365*x83 + x375*y + x382*x48 + x401*x48 - x403*x5 + x404*y);
      basis_lapl_y_eval[ipt + 10*npts] = x0*(60.0*x*x125*x239 + x*x382*x50 + x123*x154 + x125*x155*x405 + x125*x284*x405 + x13*x378 + x13*x406 + x192*x211 + x269*x315 + x322*x87);
      // Evaluate Laplacian gradient of bfn (dz)
      basis_lapl_z_eval[ipt + 0*npts] = x0*(x10*x407*y + x221*x242 + x318*x56 + x355*x54 + x380*x56 + x396*x59 + x398*x92);
      basis_lapl_z_eval[ipt + 1*npts] = x127*(x16*x407*z + x21*x331 + x21*x388 + 3.0*x286 + x306 + x408*x59 + x408*x97);
      basis_lapl_z_eval[ipt + 2*npts] = x17*(x104*x398 + x165*x410 + 32.0*x197*x409 + x221*x247 + x25*x407*y + x278*x329 + x328*x411 + x330*x411 + x338*x56 + x389*x56 + x403*x59*z + 192.0*x409*x5 + x412*x65);
      basis_lapl_z_eval[ipt + 3*npts] = x132*(x107*x408 + x135*x155 + x135*x239 + 3.0*x135*x284 + x21*x342 + x21*x392 + x22*x361 + x30*x407*z + x341 + x391 + x408*x67);
      basis_lapl_z_eval[ipt + 4*npts] = x31*(x111*x398 + x171*x410 + x221*x252 + x274*x282 + 72.0*x291*x409 + x308*x413 + x308*x414 + x345*x56 + x347*x415 + x394*x56 - 96.0*x409*x5 + x412*x69 + x416*y);
      basis_lapl_z_eval[ipt + 5*npts] = 30.0*radial_eval*x30 + 15.0*x118*x276 + 6.0*x143*x297 + 0.125*x148*x155 + 0.125*x148*x239 + 0.375*x148*x284 + 0.125*x176*x194*x22 + 0.125*x194*x22*x257 + 0.125*x21*x356 + 0.125*x21*x397 + 0.125*x223*x359 + 0.125*x223*x393 + 15.0*x224*x55 + 0.125*x225*x354 + 0.125*x277*x376 + 0.125*x38*x407*z;
      basis_lapl_z_eval[ipt + 6*npts] = x31*(24.0*x*x415 + x*x416 + x110*x412 + x178*x221 + x218*x227 + x251*x418 + 72.0*x291*x417 + x345*x93 + x358*x77 + x394*x93 + x413*x419 + x414*x419 - 96.0*x417*x7);
      basis_lapl_z_eval[ipt + 7*npts] = x40*(x150*x155 + x150*x239 + 3.0*x150*x284 + x166*x231 + x182*x201*x22 + x188 + x201*x22*x261 + x21*x366 + x21*x400 + 12.0*x216*x299 + x232*x390 + x260*x280 + x281*x362 + x407*x43*z + x62*(x181 + x420) - x62*(x420 - x95));
      basis_lapl_z_eval[ipt + 8*npts] = x17*(x*x407*x45 + x122*x412 + x185*x221 + 32.0*x197*x417 + x220*x386 + x264*x418 + x358*x82 + x369*x93 + x373*x97*z + x385*x421 + x387*x421 + x402*x93 - 192.0*x417*x7);
      basis_lapl_z_eval[ipt + 9*npts] = x46*(x128*x323 + x21*x374 + x21*x404 + x22*x267*x323 + x238*x390 + x283*x362 + 3.0*x302 + x313 + x407*x48*z);
      basis_lapl_z_eval[ipt + 10*npts] = x0*(x*x407*x50 + x125*x355 + x192*x221 + x269*x352 + x358*x87 + x377*x93 + x406*x93);




#if 0
      // Evaluate the angular part of bfn



      double ang_eval_0;
      double ang_eval_1;
      double ang_eval_2;
      double ang_eval_3;


      ang_eval_0 = radial_eval*x1*x10;
      ang_eval_1 = x12*x14*x16;
      ang_eval_2 = radial_eval*x18*x25;
      ang_eval_3 = x14*x27*x30;
      basis_eval[ipt + 0*npts] = ang_eval_0;
      basis_eval[ipt + 1*npts] = ang_eval_1;
      basis_eval[ipt + 2*npts] = ang_eval_2;
      basis_eval[ipt + 3*npts] = ang_eval_3;

      ang_eval_0 = radial_eval*x32*x36;
      ang_eval_1 = radial_eval*x37*x38;
      ang_eval_2 = radial_eval*x36*x39;
      ang_eval_3 = radial_eval*x41*x43;
      basis_eval[ipt + 4*npts] = ang_eval_0;
      basis_eval[ipt + 5*npts] = ang_eval_1;
      basis_eval[ipt + 6*npts] = ang_eval_2;
      basis_eval[ipt + 7*npts] = ang_eval_3;

      ang_eval_0 = radial_eval*x44*x45;
      ang_eval_1 = radial_eval*x47*x48;
      ang_eval_2 = radial_eval*x49*x50;
      basis_eval[ipt + 8*npts] = ang_eval_0;
      basis_eval[ipt + 9*npts] = ang_eval_1;
      basis_eval[ipt + 10*npts] = ang_eval_2;


      double dang_eval_x_0, dang_eval_y_0, dang_eval_z_0;
      double dang_eval_x_1, dang_eval_y_1, dang_eval_z_1;
      double dang_eval_x_2, dang_eval_y_2, dang_eval_z_2;
      double dang_eval_x_3, dang_eval_y_3, dang_eval_z_3;

      dang_eval_x_0 = x1*(x10*x55 + x51*x54);
      dang_eval_y_0 = x0*(radial_eval*x92 + radial_eval_alpha*x88);
      dang_eval_z_0 = x10*x118*x126;
      dang_eval_x_1 = x57*x61;
      dang_eval_y_1 = x94*x99;
      dang_eval_z_1 = x127*(radial_eval*x16 + radial_eval_alpha*x128);
      dang_eval_x_2 = x18*(x25*x55 + x62*x65);
      dang_eval_y_2 = x17*(radial_eval*x104 + radial_eval_alpha*x100);
      dang_eval_z_2 = x129*(radial_eval_alpha*x25 + x131);
      dang_eval_x_3 = x66*(radial_eval*x67 + radial_eval_alpha*x30*x5);
      dang_eval_y_3 = x105*(radial_eval*x107 + radial_eval_alpha*x30*x7);
      dang_eval_z_3 = x132*(radial_eval*x135 + radial_eval_alpha*x22*x30);
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

      dang_eval_x_0 = x32*(x36*x55 + x62*x69);
      dang_eval_y_0 = x31*(radial_eval*x111 + radial_eval_alpha*x108);
      dang_eval_z_0 = x144*x32;
      dang_eval_x_1 = x37*(x38*x55 + x51*x75);
      dang_eval_y_1 = x37*(x117*x51 + x118*x38);
      dang_eval_z_1 = 0.125*radial_eval*x148 + 0.125*radial_eval_alpha*x145;
      dang_eval_x_2 = x31*(radial_eval*x77 + radial_eval_alpha*x76);
      dang_eval_y_2 = x39*(x110*x62 + x118*x36);
      dang_eval_z_2 = x144*x39;
      dang_eval_x_3 = x41*(x43*x55 + x62*x79);
      dang_eval_y_3 = x41*(x118*x43 + x119*x62);
      dang_eval_z_3 = x40*(radial_eval*x150 + radial_eval_alpha*x149);
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

      dang_eval_x_0 = x17*(radial_eval*x82 + radial_eval_alpha*x80);
      dang_eval_y_0 = x44*(x118*x45 + x122*x62);
      dang_eval_z_0 = x151*(radial_eval_alpha*x45 + x152);
      dang_eval_x_1 = x47*x84;
      dang_eval_y_1 = x124*x47;
      dang_eval_z_1 = x46*(radial_eval*x48 + radial_eval_alpha*x153);
      dang_eval_x_2 = x0*(radial_eval*x87 + radial_eval_alpha*x85);
      dang_eval_y_2 = x49*(x118*x50 + x125*x51);
      dang_eval_z_2 = x126*x50*x55;
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
