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

#include <memory>
#include <vector>

#include <exchcxx/xc_functional.hpp>
#include <integratorxx/quadrature.hpp>
#include <integratorxx/batch/spherical_micro_batcher.hpp>

#include <gauxc/named_type.hpp>

namespace GauXC {

using functional_type = ExchCXX::XCFunctional;

// Inter-particle kernels such as EPC expect the electronic density first.
struct MultiParticlePairFunctional {
  size_t electron = 0;
  size_t particle = 0;
  std::vector<std::shared_ptr<functional_type>> functionals;
};

struct MultiParticleFunctionalSpec {
  std::vector<std::vector<std::shared_ptr<functional_type>>> intra_functionals;
  std::vector<MultiParticlePairFunctional> inter_functionals;
};

// Dynamic plan requested from a static multiparticle functional setup.
// Indices refer to the corresponding entries in MultiParticleFunctionalSpec.
struct MultiParticleXCPlan {
  std::vector<size_t> active_intra;
  std::vector<size_t> active_inter;
  std::vector<size_t> vxc_targets;
};

//using quadrature_type = IntegratorXX::QuadratureBase<
//  std::vector<std::array<double,3>>,
//  std::vector<double>
//>;
using quadrature_type = IntegratorXX::SphericalQuadratureBase<
  std::vector<std::array<double,3>>,
  std::vector<double>
>;

using batcher_type = IntegratorXX::SphericalMicroBatcher<
  typename quadrature_type::point_container,
  typename quadrature_type::weight_container
>;

}

#include <gauxc/enums.hpp>
