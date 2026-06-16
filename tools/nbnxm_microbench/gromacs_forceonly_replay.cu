#include <cuda_runtime.h>

#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "nbnxm_microbench_snapshot.h"

#include "gmxpre.h"

#include "config.h"

#include "gromacs/nbnxm/gpu_types_common.h"

#include "gromacs/nbnxm/cuda/nbnxm_cuda_kernel_utils.cuh"

#define EL_EWALD_ANA
#define CALC_ENERGIES
#define NB_KERNEL_FUNC_NAME(x, ...) x##_ElecEw_VdwLJ##__VA_ARGS__
#include "gromacs/nbnxm/cuda/nbnxm_cuda_kernel.cuh"
#undef NB_KERNEL_FUNC_NAME
#undef CALC_ENERGIES
#undef EL_EWALD_ANA

#define EL_EWALD_ANA
#define LJ_COMB_GEOM
#define CALC_ENERGIES
#define NB_KERNEL_FUNC_NAME(x, ...) x##_ElecEw_VdwLJCombGeom##__VA_ARGS__
#include "gromacs/nbnxm/cuda/nbnxm_cuda_kernel.cuh"
#undef NB_KERNEL_FUNC_NAME
#undef CALC_ENERGIES
#undef LJ_COMB_GEOM
#undef EL_EWALD_ANA

#define EL_EWALD_ANA
#define NB_KERNEL_FUNC_NAME(x, ...) x##_ElecEw_VdwLJ##__VA_ARGS__
#include "gromacs/nbnxm/cuda/nbnxm_cuda_kernel.cuh"
#undef NB_KERNEL_FUNC_NAME
#undef EL_EWALD_ANA

#define EL_EWALD_ANA
#define LJ_COMB_GEOM
#define NB_KERNEL_FUNC_NAME(x, ...) x##_ElecEw_VdwLJCombGeom##__VA_ARGS__
#include "gromacs/nbnxm/cuda/nbnxm_cuda_kernel.cuh"
#undef NB_KERNEL_FUNC_NAME
#undef LJ_COMB_GEOM
#undef EL_EWALD_ANA

namespace
{

using nbnxm_microbench::Float2POD;
using nbnxm_microbench::Float4POD;
using nbnxm_microbench::GromacsCjPackedPOD;
using nbnxm_microbench::GromacsExclPOD;
using nbnxm_microbench::GromacsPairlistSnapshot;
using nbnxm_microbench::GromacsSciPOD;

template <typename T>
inline void CheckCuda(T code, const char* what)
{
    if (code != cudaSuccess)
    {
        std::fprintf(stderr, "%s failed: %s\n", what,
                     cudaGetErrorString(code));
        std::exit(1);
    }
}

static Float4 ToGmxFloat4(const Float4POD& pod)
{
    return make_float4(pod.x, pod.y, pod.z, pod.w);
}

static Float2 ToGmxFloat2(const Float2POD& pod)
{
    return make_float2(pod.x, pod.y);
}

static Float3 ToGmxFloat3(const Float4POD& pod)
{
    return Float3{ pod.x, pod.y, pod.z };
}

static gmx::nbnxn_sci_t ToGmxSci(const GromacsSciPOD& pod)
{
    gmx::nbnxn_sci_t sci = {};
    sci.sci = pod.sci;
    sci.shift = pod.shift;
    sci.cjPackedBegin = pod.cjPackedBegin;
    sci.cjPackedEnd = pod.cjPackedEnd;
    return sci;
}

static gmx::nbnxn_cj_packed_t ToGmxCjPacked(const GromacsCjPackedPOD& pod)
{
    gmx::nbnxn_cj_packed_t packed = {};
    for (int i = 0; i < gmx::c_jGroupSize; ++i)
    {
        packed.cj[i] = pod.cj[i];
    }
    for (int i = 0; i < gmx::c_clusterSplitSize; ++i)
    {
        packed.imei[i].imask = pod.imei[i].imask;
        packed.imei[i].excl_ind = pod.imei[i].excl_ind;
    }
    return packed;
}

static gmx::nbnxn_excl_t ToGmxExcl(const GromacsExclPOD& pod)
{
    gmx::nbnxn_excl_t excl = {};
    for (int i = 0; i < gmx::c_exclSize; ++i)
    {
        excl.pair[i] = pod.pair[i];
    }
    return excl;
}

static size_t GromacsForceOnlySharedMemoryBytes(gmx::VdwType vdwType)
{
    return gmx::c_superClusterSize * gmx::c_clusterSize * sizeof(float4)
           + gmx::c_clusterSplitSize * gmx::c_jGroupSize * sizeof(int)
           + gmx::c_superClusterSize * gmx::c_clusterSize
                     * ((vdwType == gmx::VdwType::CutCombGeom)
                                ? sizeof(float2)
                                : sizeof(int))
           + sizeof(int);
}

} // namespace

void RunGromacsProduction(const GromacsPairlistSnapshot& snapshot,
                          int warmup, int iters,
                          const char* snapshotLabel)
{
    const auto& header = snapshot.header;
    const auto expectedElecType =
        static_cast<uint32_t>(gmx::ElecType::EwaldAna);
    const auto expectedVdwCut = static_cast<uint32_t>(gmx::VdwType::Cut);
    const auto expectedVdwCombGeom =
        static_cast<uint32_t>(gmx::VdwType::CutCombGeom);
    if (header.use_prune_kernel != 0u)
    {
        std::fprintf(stderr,
                     "gmx replay currently supports non-prune steady-state snapshots only\n");
        std::exit(1);
    }
    if (header.num_threads_z != 1u)
    {
        std::fprintf(stderr,
                     "gmx replay currently requires num_threads_z=1, got %u\n",
                     header.num_threads_z);
        std::exit(1);
    }
    const bool useLjCombGeom = (header.vdw_type == expectedVdwCombGeom);
    if (header.elec_type != expectedElecType
        || (header.vdw_type != expectedVdwCut
            && header.vdw_type != expectedVdwCombGeom))
            {
                std::fprintf(stderr,
                     "gmx replay currently supports ElecEw/VdwLJ or ElecEw/VdwLJCombGeom only, got elec=%u vdw=%u\n",
                     header.elec_type, header.vdw_type);
        std::exit(1);
    }
    const bool computeEnergy = (header.compute_energy != 0u);
    const bool computeVirial = (header.compute_virial != 0u);

    std::vector<Float4> xq(snapshot.sorted_xq.size());
    for (size_t i = 0; i < snapshot.sorted_xq.size(); ++i)
    {
        xq[i] = ToGmxFloat4(snapshot.sorted_xq[i]);
    }

    std::vector<int> atomTypes(snapshot.sorted_lj_type.begin(),
                               snapshot.sorted_lj_type.end());

    std::vector<Float2> ljComb(snapshot.sorted_lj_comb.size());
    for (size_t i = 0; i < snapshot.sorted_lj_comb.size(); ++i)
    {
        ljComb[i] = ToGmxFloat2(snapshot.sorted_lj_comb[i]);
    }

    std::vector<Float2> nbfp(snapshot.lj_ab.size());
    for (size_t i = 0; i < snapshot.lj_ab.size(); ++i)
    {
        nbfp[i] = ToGmxFloat2(snapshot.lj_ab[i]);
    }

    std::vector<Float3> shiftVec(header.shiftvec.size());
    for (size_t i = 0; i < header.shiftvec.size(); ++i)
    {
        shiftVec[i] = ToGmxFloat3(header.shiftvec[i]);
    }

    std::vector<gmx::nbnxn_sci_t> sci(snapshot.sci.size());
    for (size_t i = 0; i < snapshot.sci.size(); ++i)
    {
        sci[i] = ToGmxSci(snapshot.sci[i]);
    }

    std::vector<gmx::nbnxn_cj_packed_t> cjPacked(snapshot.cjpacked.size());
    for (size_t i = 0; i < snapshot.cjpacked.size(); ++i)
    {
        cjPacked[i] = ToGmxCjPacked(snapshot.cjpacked[i]);
    }

    std::vector<gmx::nbnxn_excl_t> excl(snapshot.excl.size());
    for (size_t i = 0; i < snapshot.excl.size(); ++i)
    {
        excl[i] = ToGmxExcl(snapshot.excl[i]);
    }

    gmx::NBAtomDataGpu atdat = {};
    atdat.numAtoms = static_cast<int>(header.total_atom_numbers);
    atdat.numAtomsLocal = static_cast<int>(header.local_atom_numbers);
    atdat.numAtomsAlloc = atdat.numAtoms;
    atdat.numTypes = static_cast<int>(header.num_types);

    gmx::NBParamGpu nbparam = {};
    nbparam.elecType = static_cast<gmx::ElecType>(header.elec_type);
    nbparam.vdwType = static_cast<gmx::VdwType>(header.vdw_type);
    nbparam.epsfac = header.epsfac;
    nbparam.ewald_beta = header.pme_beta;
    nbparam.rcoulomb_sq = header.cutoff * header.cutoff;
    nbparam.rvdw_sq = header.cutoff * header.cutoff;
    nbparam.useDynamicPruning = false;

    alignas(gmx::GpuPairlist) unsigned char plistStorage[sizeof(gmx::GpuPairlist)];
    std::memset(plistStorage, 0, sizeof(plistStorage));
    auto* plist = reinterpret_cast<gmx::GpuPairlist*>(plistStorage);
    plist->numAtomsPerCluster = static_cast<int>(header.cluster_size);
    plist->numSci = static_cast<int>(snapshot.sci.size());
    plist->numPackedJClusters = static_cast<int>(snapshot.cjpacked.size());
    plist->numExcl = static_cast<int>(snapshot.excl.size());
    plist->sorting.nsciSorted = plist->numSci;

    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&atdat.xq),
                         xq.size() * sizeof(Float4)),
              "cudaMalloc(xq)");
    CheckCuda(cudaMemcpy(atdat.xq, xq.data(),
                         xq.size() * sizeof(Float4),
                         cudaMemcpyHostToDevice),
              "cudaMemcpy(xq)");

    if (useLjCombGeom)
    {
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&atdat.ljComb),
                             ljComb.size() * sizeof(Float2)),
                  "cudaMalloc(ljComb)");
        CheckCuda(cudaMemcpy(atdat.ljComb, ljComb.data(),
                             ljComb.size() * sizeof(Float2),
                             cudaMemcpyHostToDevice),
                  "cudaMemcpy(ljComb)");
    }
    else
    {
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&atdat.atomTypes),
                             atomTypes.size() * sizeof(int)),
                  "cudaMalloc(atomTypes)");
        CheckCuda(cudaMemcpy(atdat.atomTypes, atomTypes.data(),
                             atomTypes.size() * sizeof(int),
                             cudaMemcpyHostToDevice),
                  "cudaMemcpy(atomTypes)");
    }

    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&atdat.shiftVec),
                         shiftVec.size() * sizeof(Float3)),
              "cudaMalloc(shiftVec)");
    CheckCuda(cudaMemcpy(atdat.shiftVec, shiftVec.data(),
                         shiftVec.size() * sizeof(Float3),
                         cudaMemcpyHostToDevice),
              "cudaMemcpy(shiftVec)");

    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&atdat.f),
                         xq.size() * sizeof(Float3)),
              "cudaMalloc(f)");
    if (computeVirial)
    {
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&atdat.fShift),
                             shiftVec.size() * sizeof(Float3)),
                  "cudaMalloc(fShift)");
        CheckCuda(cudaMemset(atdat.fShift, 0,
                             shiftVec.size() * sizeof(Float3)),
                  "cudaMemset(fShift)");
    }
    if (computeEnergy)
    {
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&atdat.eLJ),
                             sizeof(float)),
                  "cudaMalloc(eLJ)");
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&atdat.eElec),
                             sizeof(float)),
                  "cudaMalloc(eElec)");
        CheckCuda(cudaMemset(atdat.eLJ, 0, sizeof(float)),
                  "cudaMemset(eLJ)");
        CheckCuda(cudaMemset(atdat.eElec, 0, sizeof(float)),
                  "cudaMemset(eElec)");
    }

    if (!useLjCombGeom)
    {
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&nbparam.nbfp),
                             nbfp.size() * sizeof(Float2)),
                  "cudaMalloc(nbfp)");
        CheckCuda(cudaMemcpy(nbparam.nbfp, nbfp.data(),
                             nbfp.size() * sizeof(Float2),
                             cudaMemcpyHostToDevice),
                  "cudaMemcpy(nbfp)");
    }

    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&plist->sorting.sciSorted),
                         sci.size() * sizeof(gmx::nbnxn_sci_t)),
              "cudaMalloc(sciSorted)");
    CheckCuda(cudaMemcpy(plist->sorting.sciSorted, sci.data(),
                         sci.size() * sizeof(gmx::nbnxn_sci_t),
                         cudaMemcpyHostToDevice),
              "cudaMemcpy(sciSorted)");

    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&plist->cjPacked),
                         cjPacked.size() * sizeof(gmx::nbnxn_cj_packed_t)),
              "cudaMalloc(cjPacked)");
    CheckCuda(cudaMemcpy(plist->cjPacked, cjPacked.data(),
                         cjPacked.size() * sizeof(gmx::nbnxn_cj_packed_t),
                         cudaMemcpyHostToDevice),
              "cudaMemcpy(cjPacked)");

    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&plist->excl),
                         excl.size() * sizeof(gmx::nbnxn_excl_t)),
              "cudaMalloc(excl)");
    CheckCuda(cudaMemcpy(plist->excl, excl.data(),
                         excl.size() * sizeof(gmx::nbnxn_excl_t),
                         cudaMemcpyHostToDevice),
              "cudaMemcpy(excl)");

    const dim3 block(gmx::c_clusterSize, gmx::c_clusterSize, 1);
    const dim3 grid(plist->numSci, 1, 1);
    const size_t sharedMemoryBytes =
            GromacsForceOnlySharedMemoryBytes(static_cast<gmx::VdwType>(header.vdw_type));
    const void* kernel = nullptr;
    if (useLjCombGeom)
    {
        kernel = computeEnergy
                     ? reinterpret_cast<void*>(
                           gmx::nbnxn_kernel_ElecEw_VdwLJCombGeom_VF_cuda)
                     : reinterpret_cast<void*>(
                           gmx::nbnxn_kernel_ElecEw_VdwLJCombGeom_F_cuda);
    }
    else
    {
        kernel = computeEnergy
                     ? reinterpret_cast<void*>(
                           gmx::nbnxn_kernel_ElecEw_VdwLJ_VF_cuda)
                     : reinterpret_cast<void*>(gmx::nbnxn_kernel_ElecEw_VdwLJ_F_cuda);
    }

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CheckCuda(cudaEventCreate(&start), "cudaEventCreate(start)");
    CheckCuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");

    auto launchKernel = [&]() {
        CheckCuda(cudaMemset(atdat.f, 0, xq.size() * sizeof(Float3)),
                  "cudaMemset(f)");
        if (atdat.fShift != nullptr)
        {
            CheckCuda(cudaMemset(atdat.fShift, 0,
                                 shiftVec.size() * sizeof(Float3)),
                      "cudaMemset(fShift)");
        }
        if (computeEnergy)
        {
            CheckCuda(cudaMemset(atdat.eLJ, 0, sizeof(float)),
                      "cudaMemset(eLJ)");
            CheckCuda(cudaMemset(atdat.eElec, 0, sizeof(float)),
                      "cudaMemset(eElec)");
        }
        void* kernelArgs[] = { &atdat, &nbparam, plist,
                               const_cast<bool*>(&computeVirial) };
        CheckCuda(cudaLaunchKernel(
                      kernel,
                      grid,
                      block,
                      kernelArgs,
                      sharedMemoryBytes,
                      nullptr),
                  "cudaLaunchKernel(gromacs_forceonly)");
    };

    for (int iter = 0; iter < warmup; ++iter)
    {
        launchKernel();
    }
    CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(warmup)");

    CheckCuda(cudaEventRecord(start), "cudaEventRecord(start)");
    for (int iter = 0; iter < iters; ++iter)
    {
        launchKernel();
    }
    CheckCuda(cudaEventRecord(stop), "cudaEventRecord(stop)");
    CheckCuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");

    float elapsedMs = 0.0f;
    CheckCuda(cudaEventElapsedTime(&elapsedMs, start, stop),
              "cudaEventElapsedTime");

    float energyLj = 0.0f;
    float energyElec = 0.0f;
    if (computeEnergy)
    {
        CheckCuda(cudaMemcpy(&energyLj, atdat.eLJ, sizeof(float),
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy(eLJ)");
        CheckCuda(cudaMemcpy(&energyElec, atdat.eElec, sizeof(float),
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy(eElec)");
    }

    if (computeEnergy)
    {
        std::printf("kernel=gromacs_fulloutput_warp_record snapshot=%s "
                    "avg_ms=%.6f iters=%d sci=%zu cjpacked=%zu excl=%zu "
                    "atoms=%zu compute_energy=%u compute_virial=%u "
                    "energy_lj=%.6e energy_el=%.6e\n",
                    snapshotLabel,
                    elapsedMs / static_cast<float>(iters),
                    iters,
                    snapshot.sci.size(),
                    snapshot.cjpacked.size(),
                    snapshot.excl.size(),
                    snapshot.sorted_xq.size(),
                    header.compute_energy,
                    header.compute_virial,
                    energyLj,
                    energyElec);
    }
    else
    {
        std::printf("kernel=gromacs_forceonly_warp_record snapshot=%s "
                    "avg_ms=%.6f iters=%d sci=%zu cjpacked=%zu excl=%zu "
                    "atoms=%zu compute_energy=%u compute_virial=%u\n",
                    snapshotLabel,
                    elapsedMs / static_cast<float>(iters),
                    iters,
                    snapshot.sci.size(),
                    snapshot.cjpacked.size(),
                    snapshot.excl.size(),
                    snapshot.sorted_xq.size(),
                    header.compute_energy,
                    header.compute_virial);
    }

    CheckCuda(cudaEventDestroy(start), "cudaEventDestroy(start)");
    CheckCuda(cudaEventDestroy(stop), "cudaEventDestroy(stop)");

    CheckCuda(cudaFree(plist->excl), "cudaFree(excl)");
    CheckCuda(cudaFree(plist->cjPacked), "cudaFree(cjPacked)");
    CheckCuda(cudaFree(plist->sorting.sciSorted), "cudaFree(sciSorted)");
    CheckCuda(cudaFree(nbparam.nbfp), "cudaFree(nbfp)");
    CheckCuda(cudaFree(atdat.eElec), "cudaFree(eElec)");
    CheckCuda(cudaFree(atdat.eLJ), "cudaFree(eLJ)");
    CheckCuda(cudaFree(atdat.fShift), "cudaFree(fShift)");
    CheckCuda(cudaFree(atdat.f), "cudaFree(f)");
    CheckCuda(cudaFree(atdat.shiftVec), "cudaFree(shiftVec)");
    CheckCuda(cudaFree(atdat.ljComb), "cudaFree(ljComb)");
    CheckCuda(cudaFree(atdat.atomTypes), "cudaFree(atomTypes)");
    CheckCuda(cudaFree(atdat.xq), "cudaFree(xq)");
}
