#pragma once

#include "neighbor_list/contract/types.h"
#include "neighbor_list/contract/traversal.cuh"

void Launch_Clustered_Gmxpacked_Record_Stream_Source_Materialize_From_Gmxpacked(
    int sci_numbers, int builder_block_size,
    const LJ_CLUSTERED_GMXPACKED_SCI* gmxpacked_sci,
    const LJ_CLUSTERED_GMXPACKED_CJ* gmxpacked_cjpacked,
    const LJ_CLUSTERED_GMXPACKED_EXCLUSION* exclusion_entries,
    int source_capacity,
    LJ_CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE* sources,
    int* source_cursor, int* overflow_rows);

void Launch_Clustered_Gmxpacked_Record_Stream_Inner_Active_Count_Probe(
    int source_rows, int builder_block_size,
    const LJ_CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE* sources,
    const int* permutation, const int* cluster_offsets,
    const int* super_cluster_offsets, const unsigned int* cluster_local_masks,
    const VECTOR* cluster_centers, const VECTOR* crd, LTMatrix3 cell,
    LTMatrix3 rcell, float cutoff_sq, int* active_flags,
    unsigned int* active_imasks_by_source);

void Launch_Clustered_Gmxpacked_Record_Stream_Inner_Active_Fill_Probe(
    int source_rows, int builder_block_size,
    const LJ_CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE* sources,
    const int* permutation, const int* cluster_offsets,
    const int* super_cluster_offsets, const unsigned int* cluster_local_masks,
    const VECTOR* cluster_centers, const VECTOR* crd, LTMatrix3 cell,
    LTMatrix3 rcell, float cutoff_sq, const int* active_offsets,
    LJ_CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE* active_sources);

void Launch_Clustered_Gmxpacked_Record_Stream_Inner_Active_Fill_Cached_Probe(
    int source_rows, int builder_block_size,
    const LJ_CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE* sources,
    const unsigned int* active_imasks_by_source, const int* active_offsets,
    LJ_CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE* active_sources);
