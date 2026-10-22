#/bin/bash

if [ ! -d kernels ]
then
  mkdir kernels
fi

if [ ! -d kernels/collocation ]
then
  mkdir -p kernels/collocation
fi

export CUDA_PREFIX=$PWD/../cuda/kernels
export HIP_PREFIX=$PWD/kernels

# Generate collocation kernels
hipify-perl $CUDA_PREFIX/collocation/collocation_angular_cartesian.hpp > \
            $HIP_PREFIX/collocation/collocation_angular_cartesian.hpp
hipify-perl $CUDA_PREFIX/collocation/collocation_angular_spherical_unnorm.hpp > \
            $HIP_PREFIX/collocation/collocation_angular_spherical_unnorm.hpp
hipify-perl $CUDA_PREFIX/collocation/collocation_device_constants.hpp > \
            $HIP_PREFIX/collocation/collocation_device_constants.hpp
hipify-perl $CUDA_PREFIX/collocation_masked_combined_kernels.hpp > \
            $HIP_PREFIX/collocation_masked_combined_kernels.hpp
hipify-perl $CUDA_PREFIX/collocation_masked_kernels.hpp > \
            $HIP_PREFIX/collocation_masked_kernels.hpp
# hipify-perl $CUDA_PREFIX/collocation_device.hpp > \
#            $HIP_PREFIX/collocation_device.hpp
hipify-perl $CUDA_PREFIX/collocation_device.cu > \
            $HIP_PREFIX/collocation_device.hip


# Generate Weights Kernels
hipify-perl $CUDA_PREFIX/grid_to_center.hpp > $HIP_PREFIX/grid_to_center.hpp
hipify-perl $CUDA_PREFIX/grid_to_center.cu  > $HIP_PREFIX/grid_to_center.hip
hipify-perl $CUDA_PREFIX/cuda_ssf_1d.hpp > $HIP_PREFIX/hip_ssf_1d.hpp
hipify-perl $CUDA_PREFIX/cuda_ssf_1d.cu  > $HIP_PREFIX/hip_ssf_1d.hip


# cuBLAS -> hipBLAS
# hipify-perl $CUDA_PREFIX/cublas_extensions.hpp > $HIP_PREFIX/hipblas_extensions.hpp
hipify-perl $CUDA_PREFIX/cublas_extensions.cu > $HIP_PREFIX/hipblas_extensions.hip

# Z Matrix
# hipify-perl $CUDA_PREFIX/zmat_vxc.hpp > $HIP_PREFIX/zmat_vxc.hpp
hipify-perl $CUDA_PREFIX/zmat_vxc.cu > $HIP_PREFIX/zmat_vxc.hip

# new kernels
hipify-perl $CUDA_PREFIX/cuda_inc_potential.cu > $HIP_PREFIX/hip_inc_potential.hip

hipify-perl $CUDA_PREFIX/increment_exc_grad.cu > $HIP_PREFIX/increment_exc_grad.hip
# hipify-perl $CUDA_PREFIX/increment_exc_grad.hpp > $HIP_PREFIX/increment_exc_grad.hpp

hipify-perl $CUDA_PREFIX/exx_ek_screening_bfn_stats.cu > $HIP_PREFIX/exx_ek_screening_bfn_stats.hip

hipify-perl $CUDA_PREFIX/zmat_fxc.cu > $HIP_PREFIX/zmat_fxc.hip
# hipify-perl $CUDA_PREFIX/zmat_fxc.hpp > $HIP_PREFIX/zmat_fxc.hpp

hipify-perl $CUDA_PREFIX/symmetrize_mat.cu > $HIP_PREFIX/symmetrize_mat.hip
hipify-perl $CUDA_PREFIX/pack_submat.cu > $HIP_PREFIX/pack_submat.hip
hipify-perl $CUDA_PREFIX/uvvars.cu > $HIP_PREFIX/uvvars.hip

hipify-perl $CUDA_PREFIX/uvvars_lda.hpp > $HIP_PREFIX/uvvars_lda.hpp
hipify-perl $CUDA_PREFIX/uvvars_gga.hpp > $HIP_PREFIX/uvvars_gga.hpp
hipify-perl $CUDA_PREFIX/uvvars_mgga.hpp > $HIP_PREFIX/uvvars_mgga.hpp

hipify-perl $CUDA_PREFIX/cuda_extensions.hpp > $HIP_PREFIX/hip_extensions.hpp
hipify-perl $CUDA_PREFIX/collocation_shell_to_task_kernels.hpp > $HIP_PREFIX/collocation_shell_to_task_kernels.hpp

hipify-perl $CUDA_PREFIX/cuda_extensions.hpp > $HIP_PREFIX/hip_extensions.hpp

hipify-perl $CUDA_PREFIX/../scheme1_cutlass_base.cxx> $HIP_PREFIX/../scheme1_cutlass_base.cxx
hipify-perl $CUDA_PREFIX/../scheme1_cutlass_base.hpp > $HIP_PREFIX/../scheme1_cutlass_base.hpp
hipify-perl $CUDA_PREFIX/../scheme1_cutlass_data_base.cxx > $HIP_PREFIX/../scheme1_cutlass_data_base.cxx

hipify-perl $CUDA_PREFIX/../cuda_aos_scheme1.cxx > $HIP_PREFIX/../hip_aos_scheme1.cxx
hipify-perl $CUDA_PREFIX/../cuda_aos_scheme1.hpp > $HIP_PREFIX/../hip_aos_scheme1.hpp
hipify-perl $CUDA_PREFIX/../cuda_aos_scheme1_data.cxx > $HIP_PREFIX/../hip_aos_scheme1_data.cxx
hipify-perl $CUDA_PREFIX/../cuda_aos_scheme1_weights.cu > $HIP_PREFIX/../hip_aos_scheme1_weights.hip
hipify-perl $CUDA_PREFIX/../cuda_aos_scheme1_weights.hpp > $HIP_PREFIX/../hip_aos_scheme1_weights.hpp
hipify-perl $CUDA_PREFIX/../xc_functional_eval_wrapper.cxx > $HIP_PREFIX/../xc_functional_eval_wrapper.cxx

for file in $CUDA_PREFIX/collocation/collocation_shell_to_task_kernels*.hpp; do
  [ -e "$file" ] || continue
  bname=$(basename "$file")
  hipify-perl "$file" > $HIP_PREFIX/collocation/"$bname"
done

sed -i "" -e "s/cuda/hip/g" kernels/{,*/}*.hpp *.{cxx,hpp}
sed -i "" -e "s/cuda/hip/g" kernels/*.hip
sed -i "" -e "s/CUDA/HIP/g" kernels/*.hip
sed -i "" -e "s/cublas/hipblas/g" kernels/*.hip
sed -i "" -e "s/CUBLAS/HIPBLAS/g" kernels/*.hip
sed -i "" -e "s/register //g" kernels/*.hip

#sed -i -e "s/Cuda/Hip/g" *.{cxx,hpp}
