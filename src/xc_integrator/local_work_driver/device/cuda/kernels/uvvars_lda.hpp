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
#include "device_specific/cuda_device_constants.hpp"
#include "device_specific/cuda_util.hpp"
#include "device/xc_device_data.hpp"

#include <stdio.h>

namespace GauXC {

template <bool trial, density_id den_select>
__global__ void eval_vvar_lda_kern( size_t        ntasks,
                                    XCDeviceTask* tasks_device) {

  const int batch_idx = blockIdx.z;
  if( batch_idx >= ntasks ) return;

  auto& task = tasks_device[ batch_idx ];

  const auto npts            = task.npts;
  const auto nbf             = task.bfn_screening.nbe;

  double* den_eval_device   = nullptr;
  // use the "U" variable (+/- for UKS) even though at this point the density (S/Z) is stored
  if constexpr (trial){
    if constexpr (den_select == DEN_S) den_eval_device = task.tden_s;
    if constexpr (den_select == DEN_Z) den_eval_device = task.tden_z;
    if constexpr (den_select == DEN_Y) den_eval_device = task.tden_y;
    if constexpr (den_select == DEN_X) den_eval_device = task.tden_x;
  }else{
      if constexpr (den_select == DEN_S) den_eval_device = task.den_s;
      if constexpr (den_select == DEN_Z) den_eval_device = task.den_z;
      if constexpr (den_select == DEN_Y) den_eval_device = task.den_y;
      if constexpr (den_select == DEN_X) den_eval_device = task.den_x;
  }

  const auto* basis_eval_device = task.bf;

  const auto* den_basis_prod_device = task.zmat;

  const int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int tid_y = blockIdx.y * blockDim.y + threadIdx.y;

  register double den_reg = 0.;

  if( tid_x < nbf and tid_y < npts ) {

    const double* bf_col   = basis_eval_device     + tid_x*npts;
    const double* db_col   = den_basis_prod_device + tid_x*npts;

    den_reg = bf_col[ tid_y ]   * db_col[ tid_y ];

  }

  // Warp blocks are stored col major
  constexpr auto warp_size = cuda::warp_size;
  //constexpr auto max_warps_per_thread_block = cuda::max_warps_per_thread_block;
  den_reg = cuda::warp_reduce_sum<warp_size>( den_reg );


  if( threadIdx.x == 0 and tid_y < npts ) {
    atomicAdd( den_eval_device   + tid_y, den_reg );
  }
  
}

template <bool trial, density_id den_select>
__global__ void eval_vvar_lda_dks_kern( size_t        ntasks,
                                    XCDeviceTask* tasks_device) {

  const int batch_idx = blockIdx.z;
  if( batch_idx >= ntasks ) return;

  auto& task = tasks_device[ batch_idx ];
  // printf("eval_vvar_lda_dks_kern");
  const auto npts            = task.npts;
  const auto nbf             = task.bfn_screening.nbe;
  double xx, yy ,zz, mkxy ,mkyx, mjxz, mjzx, mizy, miyz;
  
  double SpeedOfLight  = 137.03599917700001;
  double RKB_factor = 1./(4.*SpeedOfLight*SpeedOfLight);

  double* den_eval_device   = nullptr;
  // use the "U" variable (+/- for UKS) even though at this point the density (S/Z) is stored
  if constexpr (trial){
    if constexpr (den_select == DEN_S) den_eval_device = task.tden_s;
    if constexpr (den_select == DEN_Z) den_eval_device = task.tden_z;
    if constexpr (den_select == DEN_Y) den_eval_device = task.tden_y;
    if constexpr (den_select == DEN_X) den_eval_device = task.tden_x;
  }else{
      if constexpr (den_select == DEN_S) {  //k(xy+m*yx) j(xz+m*zx) i(zy+m*yz)
        den_eval_device = task.den_s; // k: Z ; j: X ; i: Y
        xx = 1.;
        yy = 1.;
        zz = 1.;
        mkxy=-1.;
        mkyx=1.;
        mjxz=1.;
        mjzx=-1.;
        mizy=1.;
        miyz=-1.;
      }
      if constexpr (den_select == DEN_Z) {
        den_eval_device = task.den_z; // k: S ; j: X ; i: Y
        xx = -1.;
        yy = -1.;
        zz = 1.;
        mkxy=1.;
        mkyx=-1.;
        mjxz=1.;
        mjzx=1.;
        mizy=1.;
        miyz=1.;
      }
      if constexpr (den_select == DEN_Y) { 
        den_eval_device = task.den_y; // k: X ; j: S ; i: Z
        xx = -1.;
        yy = 1.;
        zz = -1.;
        mkyx=1.;
        mkyx=1.;
        mjxz=-1.;
        mjzx=1.;
        mizy=1.;
        miyz=1.;
      }
      if constexpr (den_select == DEN_X) { 
        den_eval_device = task.den_x; // k: Y ; j: Z ; i: S
        xx = 1.;
        yy = -1.;
        zz = -1.;
        mkyx=1.;
        mkyx=1.;
        mjxz=1.;
        mjzx=1.;
        mizy=-1.;
        miyz=1.;
      }
  }

  const auto* basis_eval_device = task.bf;
  const auto* dbasis_x_eval_device = task.dbfx;
  const auto* dbasis_y_eval_device = task.dbfy;
  const auto* dbasis_z_eval_device = task.dbfz;

  const auto* den_basis_LL_prod_device = task.zmat;
  const auto* den_basis_SS_xx_prod_device  = task.xmat_x;
  const auto* den_basis_SS_yy_prod_device  = task.xmat_y;
  const auto* den_basis_SS_zz_prod_device  = task.xmat_z;
  const auto* den_basis_SS_kxy_prod_device = task.xmat_k_ij;
  const auto* den_basis_SS_kyx_prod_device = task.xmat_k_ji;
  const auto* den_basis_SS_jzx_prod_device = task.xmat_j_ik;
  const auto* den_basis_SS_jxz_prod_device = task.xmat_j_ki;
  const auto* den_basis_SS_iyz_prod_device = task.xmat_i_jk;
  const auto* den_basis_SS_izy_prod_device = task.xmat_i_kj;

  const int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int tid_y = blockIdx.y * blockDim.y + threadIdx.y;

  register double den_reg = 0.;

  if( tid_x < nbf and tid_y < npts ) {

    const double* bf_col   = basis_eval_device     + tid_x*npts;
    const double* bf_x_col = dbasis_x_eval_device  + tid_x*npts;
    const double* bf_y_col = dbasis_y_eval_device  + tid_x*npts;
    const double* bf_z_col = dbasis_z_eval_device  + tid_x*npts;

    const double* db_col   = den_basis_LL_prod_device + tid_x*npts;
    den_reg = bf_col[ tid_y ]   * db_col[ tid_y ];
    
    // rho SS
    db_col   = den_basis_SS_xx_prod_device  + tid_x*npts;
    den_reg += xx*RKB_factor*(bf_x_col[ tid_y ]   * db_col[ tid_y ]);

    db_col   = den_basis_SS_yy_prod_device  + tid_x*npts;
    den_reg += yy*RKB_factor*(bf_y_col[ tid_y ]   * db_col[ tid_y ]);

    db_col   = den_basis_SS_zz_prod_device  + tid_x*npts;
    den_reg += zz*RKB_factor*(bf_z_col[ tid_y ]   * db_col[ tid_y ]);

    db_col   = den_basis_SS_kxy_prod_device + tid_x*npts;
    den_reg += mkxy*RKB_factor*(bf_y_col[ tid_y ]   * db_col[ tid_y ]);

    db_col   = den_basis_SS_kyx_prod_device + tid_x*npts;
    den_reg += mkyx*RKB_factor*(bf_x_col[ tid_y ]   * db_col[ tid_y ]);

    db_col   = den_basis_SS_jzx_prod_device + tid_x*npts;
    den_reg += mjzx*RKB_factor*(bf_x_col[ tid_y ]   * db_col[ tid_y ]);

    db_col   = den_basis_SS_jxz_prod_device + tid_x*npts;
    den_reg += mjxz*RKB_factor*(bf_z_col[ tid_y ]   * db_col[ tid_y ]);

    db_col   = den_basis_SS_iyz_prod_device + tid_x*npts;
    den_reg += miyz*RKB_factor*(bf_y_col[ tid_y ]   * db_col[ tid_y ]);

    db_col   = den_basis_SS_izy_prod_device + tid_x*npts;
    den_reg += mizy*RKB_factor*(bf_z_col[ tid_y ]   * db_col[ tid_y ]);
    // printf("eval_vvar_lda_dks_kern, den_reg %.10f\n",den_reg);
  }

  // Warp blocks are stored col major
  constexpr auto warp_size = cuda::warp_size;
  //constexpr auto max_warps_per_thread_block = cuda::max_warps_per_thread_block;
  den_reg = cuda::warp_reduce_sum<warp_size>( den_reg );

  printf("den_reg, %.10f\n", den_reg);

  if( threadIdx.x == 0 and tid_y < npts ) {
    atomicAdd( den_eval_device   + tid_y, den_reg );
    printf("den_eval_device, %.10f\n", den_eval_device   + tid_y);

  }
  
}

__global__ void eval_uvars_lda_rks_kernel( size_t ntasks, XCDeviceTask* tasks_device) {
  // eval_vvars populated uvar storage already in the case of LDA+RKS
  return;
}
__global__ void eval_tmat_lda_rks_kernel( size_t ntasks, XCDeviceTask* tasks_device) {

  const int batch_idx = blockIdx.z;
  if( batch_idx >= ntasks ) return;

  const auto& task = tasks_device[ batch_idx ];
  const auto npts  = task.npts;

  const auto* v2rho2_device  = task.v2rho2;
  const auto* weight_device  = task.weights;
  auto* tden_s_eval_device   = task.tden_s;
  auto* FXC_A_device   = task.FXC_A_s;

  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if( tid < npts ) {
    FXC_A_device[ tid ] = v2rho2_device[ tid ] * tden_s_eval_device[ tid ] * weight_device[ tid ];
  }

  return;
}


__global__ void eval_uvars_lda_uks_kernel( size_t        ntasks,
  XCDeviceTask* tasks_device ) {

  const int batch_idx = blockIdx.z;
  if( batch_idx >= ntasks ) return;

  auto& task = tasks_device[ batch_idx ];

  const auto npts            = task.npts;

  auto* den_pos_eval_device   = task.den_s;
  auto* den_neg_eval_device   = task.den_z;

  const int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if( tid < npts ) {
    const auto ps = den_pos_eval_device[ tid ];
    const auto pz = den_neg_eval_device[ tid ];
    den_pos_eval_device[ tid ] = 0.5*(ps + pz);
    den_neg_eval_device[ tid ] = 0.5*(ps - pz);
  }
}

__global__ void eval_tmat_lda_uks_kernel( size_t        ntasks,
  XCDeviceTask* tasks_device ) {

  const int batch_idx = blockIdx.z;
  if( batch_idx >= ntasks ) return;

  auto& task = tasks_device[ batch_idx ];

  const auto npts            = task.npts;

  auto* tden_s_device   = task.tden_s;
  auto* tden_z_device   = task.tden_z;
  auto* FXC_A_s_device        = task.FXC_A_s;
  auto* FXC_A_z_device        = task.FXC_A_z;
  const auto* weight_device   = task.weights;

  const auto* v2rho2_a_a_device    = task.v2rho2_a_a;
  const auto* v2rho2_a_b_device    = task.v2rho2_a_b;
  const auto* v2rho2_b_b_device    = task.v2rho2_b_b;

  const int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if( tid < npts ) {
    const auto ps = tden_s_device[ tid ];
    const auto pz = tden_z_device[ tid ];
    const auto trho_a_device = 0.5*(ps + pz);
    const auto trho_b_device = 0.5*(ps - pz);
    const auto A_a = v2rho2_a_a_device[tid] * trho_a_device + v2rho2_a_b_device[tid] * trho_b_device;
    const auto A_b = v2rho2_b_b_device[tid] * trho_b_device + v2rho2_a_b_device[tid] * trho_a_device;
    FXC_A_s_device[ tid ] = 0.5 * weight_device[ tid ] * (A_a + A_b);
    FXC_A_z_device[ tid ] = 0.5 * weight_device[ tid ] * (A_a - A_b);
  }
}

__global__ void eval_uvars_lda_gks_kernel( size_t        ntasks,
                                           XCDeviceTask* tasks_device ) {

  const int batch_idx = blockIdx.z;
  if( batch_idx >= ntasks ) return;

  auto& task = tasks_device[ batch_idx ];

  const auto npts            = task.npts;

  auto* den_z_eval_device   = task.den_s;
  auto* den_s_eval_device   = task.den_z;
  auto* den_y_eval_device   = task.den_y;
  auto* den_x_eval_device   = task.den_x;
  auto* K_z_eval_device     = task.K_z;
  auto* K_y_eval_device     = task.K_y;
  auto* K_x_eval_device     = task.K_x;
  const double dtolsq = 1e-24;  // TODO: make variable

  const int tid = blockIdx.x * blockDim.x + threadIdx.x;


  if( tid < npts ) {
    const auto ps = den_s_eval_device[ tid ];
    const auto pz = den_z_eval_device[ tid ];
    const auto py = den_y_eval_device[ tid ];
    const auto px = den_x_eval_device[ tid ];
    const auto mtemp = pz*pz + px*px + py*py;
    double mnorm = 0.;
  
    if (mtemp > dtolsq) {
      const double inv_mnorm = rsqrt(mtemp);
      mnorm = 1./inv_mnorm;
      K_z_eval_device[ tid ] = pz * inv_mnorm;
      K_y_eval_device[ tid ] = py * inv_mnorm;
      K_x_eval_device[ tid ] = px * inv_mnorm;
    }
    else {
      mnorm = (1. / 3.) * (px + py + pz);
      K_z_eval_device[ tid ] = 1. / 3.;
      K_y_eval_device[ tid ] = 1. / 3.;
      K_x_eval_device[ tid ] = 1. / 3.;
    }

    den_s_eval_device[ tid ] = 0.5*(ps + mnorm);
    den_z_eval_device[ tid ] = 0.5*(ps - mnorm);

  }
}

__global__ void eval_uvars_lda_dks_kernel( size_t        ntasks,
                                           XCDeviceTask* tasks_device ) {

  const int batch_idx = blockIdx.z;
  if( batch_idx >= ntasks ) return;

  auto& task = tasks_device[ batch_idx ];

  const auto npts            = task.npts;

  auto* den_z_eval_device   = task.den_s;
  auto* den_s_eval_device   = task.den_z;
  auto* den_y_eval_device   = task.den_y;
  auto* den_x_eval_device   = task.den_x;
  auto* K_z_eval_device     = task.K_z;
  auto* K_y_eval_device     = task.K_y;
  auto* K_x_eval_device     = task.K_x;
  const double dtolsq = 1e-24;  // TODO: make variable

  const int tid = blockIdx.x * blockDim.x + threadIdx.x;


  if( tid < npts ) {
    const auto ps = den_s_eval_device[ tid ];
    const auto pz = den_z_eval_device[ tid ];
    const auto py = den_y_eval_device[ tid ];
    const auto px = den_x_eval_device[ tid ];
    const auto mtemp = pz*pz + px*px + py*py;
    double mnorm = 0.;
    // printf("ps, %.10f pz, %.10f py, %.10f px, %.10f\n", ps, pz, py, px);
    
    if (mtemp > dtolsq) {
      const double inv_mnorm = rsqrt(mtemp);
      mnorm = 1./inv_mnorm;
      K_z_eval_device[ tid ] = pz * inv_mnorm;
      K_y_eval_device[ tid ] = py * inv_mnorm;
      K_x_eval_device[ tid ] = px * inv_mnorm;
    }
    else {
      mnorm = (1. / 3.) * (px + py + pz);
      K_z_eval_device[ tid ] = 1. / 3.;
      K_y_eval_device[ tid ] = 1. / 3.;
      K_x_eval_device[ tid ] = 1. / 3.;
    }

    den_s_eval_device[ tid ] = 0.5*(ps + mnorm);
    den_z_eval_device[ tid ] = 0.5*(ps - mnorm);
    // printf("den_s_eval_device, lda rho+ %.10f\n",0.5*(ps + mnorm));
  }
}

}
