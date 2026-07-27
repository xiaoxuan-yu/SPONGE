#include "neighbor_list/clustered_spatial_view.h"
#include "Lennard_Jones_force/clustered_lj.h"

#include <cstdio>
#include <cstring>

namespace
{
int failures = 0;

void Check(bool condition, const char* label)
{
    if (!condition)
    {
        std::fprintf(stderr, "FAIL: %s\n", label);
        failures += 1;
    }
}

void Expect_Valid(const char* label, const CLUSTERED_SPATIAL_VIEW& view,
                  const CLUSTERED_SPATIAL_VIEW_REQUIREMENTS& requirements)
{
    const char* reason = reinterpret_cast<const char*>(1);
    const bool valid =
        Clustered_Validate_Spatial_View(view, requirements, &reason);
    Check(valid, label);
    Check(reason == nullptr, "successful validation clears failure reason");
}

void Expect_Invalid(const char* label, const char* expected_reason,
                    const CLUSTERED_SPATIAL_VIEW& view,
                    const CLUSTERED_SPATIAL_VIEW_REQUIREMENTS& requirements)
{
    const char* reason = nullptr;
    const bool valid =
        Clustered_Validate_Spatial_View(view, requirements, &reason);
    Check(!valid, label);
    Check(reason != nullptr && std::strcmp(reason, expected_reason) == 0,
          expected_reason);
}

struct View_Fixture
{
    int sort_permutation[3] = {0, 1, 2};
    int cluster_offsets[2] = {0, 3};
    unsigned int cluster_valid_masks[1] = {0x7u};
    unsigned int cluster_local_masks[1] = {0x3u};
    VECTOR cluster_centers[1] = {};
    VECTOR cluster_extents[1] = {};
    int super_cluster_offsets[2] = {0, 1};
    CLUSTERED_SCI sci[1] = {};
    CLUSTERED_CJ_PACKED cjpacked[1] = {};

    CLUSTERED_SPATIAL_VIEW view = {};

    View_Fixture()
    {
        sci[0].cjpacked_end = 1;
        view.ready = true;
        view.backend = CLUSTERED_SPATIAL_BACKEND::CPU;
        view.readiness_scope =
            CLUSTERED_SPATIAL_READINESS_SCOPE::HOST_COMPLETE;
        view.provider_incarnation = 3;
        view.lease_epoch = 5;
        view.native_payload_generation = 7;
        view.gmxpacked_payload_generation = 17;
        view.source_generation = 11;
        view.geometry_generation = 13;
        view.cluster_size = kClusteredClusterSize;
        view.super_cluster_clusters = kClusteredSuperClusterClusters;
        view.local_atom_numbers = 2;
        view.direct_local_atom_numbers = 2;
        view.ghost_numbers = 1;
        view.total_atom_numbers = 3;
        view.padded_total_atom_numbers = 3;
        view.cluster_numbers = 1;
        view.super_cluster_numbers = 1;
        view.sci_numbers = 1;
        view.cjpacked_numbers = 1;
        view.exclusion_pool_numbers = 0;
        view.cached_cutoff = 10.0f;
        view.sort_permutation = sort_permutation;
        view.cluster_offsets = cluster_offsets;
        view.cluster_valid_masks = cluster_valid_masks;
        view.cluster_local_masks = cluster_local_masks;
        view.cluster_centers = cluster_centers;
        view.cluster_extents = cluster_extents;
        view.super_cluster_offsets = super_cluster_offsets;
        view.sci = sci;
        view.cjpacked = cjpacked;
    }
};

CLUSTERED_SPATIAL_VIEW_REQUIREMENTS Base_Requirements()
{
    CLUSTERED_SPATIAL_VIEW_REQUIREMENTS requirements;
    requirements.local_atom_numbers = 2;
    requirements.ghost_numbers = 1;
    requirements.cutoff = 9.0f;
    requirements.provider_incarnation = 3;
    requirements.lease_epoch = 5;
    requirements.native_payload_generation = 7;
    requirements.source_generation = 11;
    requirements.geometry_generation = 13;
    requirements.require_backend = true;
    requirements.backend = CLUSTERED_SPATIAL_BACKEND::CPU;
    requirements.require_native_payload = true;
    return requirements;
}

void Test_Validation_Contract()
{
    View_Fixture fixture;
    const auto requirements = Base_Requirements();

    // A valid native payload may have no exclusion entries.
    Check(fixture.view.exclusion_mask_pool == nullptr,
          "empty exclusion fixture must use a null pool");
    Expect_Valid("valid native view with empty exclusion pool", fixture.view,
                 requirements);

    {
        auto view = fixture.view;
        view.ready = false;
        Expect_Invalid("not-ready view",
                       "clustered spatial view is not ready", view,
                       requirements);
    }
    {
        auto stale = requirements;
        stale.provider_incarnation += 1;
        Expect_Invalid(
            "stale provider incarnation",
            "clustered spatial view provider incarnation is stale",
            fixture.view, stale);
    }
    {
        auto stale = requirements;
        stale.lease_epoch += 1;
        Expect_Invalid("stale lease epoch",
                       "clustered spatial view lease epoch is stale",
                       fixture.view, stale);
    }
    {
        auto stale = requirements;
        stale.native_payload_generation += 1;
        Expect_Invalid(
            "stale native payload generation",
            "clustered spatial view native payload generation is stale",
                       fixture.view, stale);
    }
    {
        auto stale = requirements;
        stale.source_generation += 1;
        Expect_Invalid("stale source generation",
                       "clustered spatial view source generation is stale",
                       fixture.view, stale);
    }
    {
        auto stale = requirements;
        stale.geometry_generation += 1;
        Expect_Invalid(
            "stale geometry generation after coordinate/domain change",
            "clustered spatial view geometry generation is stale",
            fixture.view, stale);
    }
    {
        auto mismatch = requirements;
        mismatch.backend = CLUSTERED_SPATIAL_BACKEND::CUDA;
        Expect_Invalid("backend mismatch",
                       "clustered spatial view backend does not match",
                       fixture.view, mismatch);
    }
    {
        auto view = fixture.view;
        view.backend = CLUSTERED_SPATIAL_BACKEND::CUDA;
        view.readiness_scope =
            CLUSTERED_SPATIAL_READINESS_SCOPE::PRODUCER_STREAM_ORDERED;
        view.producer_stream = reinterpret_cast<const void*>(1);
        auto mismatch = requirements;
        mismatch.backend = CLUSTERED_SPATIAL_BACKEND::CUDA;
        mismatch.require_same_producer_stream = true;
        mismatch.consumer_stream = reinterpret_cast<const void*>(2);
        Expect_Invalid(
            "producer stream mismatch",
            "clustered spatial view producer stream does not match", view,
            mismatch);
    }
    {
        auto mismatch = requirements;
        mismatch.local_atom_numbers += 1;
        Expect_Invalid("local atom mismatch",
                       "clustered spatial view local atom count does not match",
                       fixture.view, mismatch);
    }
    {
        auto mismatch = requirements;
        mismatch.cutoff = 10.01f;
        Expect_Invalid(
            "cutoff mismatch",
            "clustered spatial view cutoff is not a semantic superset",
            fixture.view, mismatch);
    }
    {
        auto view = fixture.view;
        view.cluster_centers = nullptr;
        Expect_Invalid(
            "missing structural pointer",
            "clustered spatial view is missing cluster_centers", view,
            requirements);
    }
    {
        auto need_shift = requirements;
        need_shift.require_pair_shift_metadata = true;
        Expect_Invalid(
            "missing pair-shift capability",
            "clustered spatial view pair-shift metadata is unavailable",
            fixture.view, need_shift);
        uint64_t pair_shift_bits[kClusteredJGroupSize] = {};
        CLUSTERED_GMXPACKED_SCI gmxpacked_sci[1] = {};
        CLUSTERED_GMXPACKED_CJ gmxpacked_cjpacked[1] = {};
        CLUSTERED_GMXPACKED_EXCLUSION gmxpacked_exclusions[1] = {};
        auto view = fixture.view;
        view.gmxpacked_sci_numbers = 1;
        view.gmxpacked_cjpacked_numbers = 1;
        view.gmxpacked_exclusion_numbers = 1;
        view.gmxpacked_sci = gmxpacked_sci;
        view.gmxpacked_cjpacked = gmxpacked_cjpacked;
        view.gmxpacked_exclusions = gmxpacked_exclusions;
        view.pair_shift_metadata_ready = true;
        view.pair_shift_payload_generation =
            view.gmxpacked_payload_generation;
        view.pair_shift_geometry_generation = view.geometry_generation;
        view.pair_shift_sci_numbers = view.gmxpacked_sci_numbers;
        view.pair_shift_cjpacked_numbers = view.gmxpacked_cjpacked_numbers;
        view.pair_shift_exclusion_numbers =
            view.gmxpacked_exclusion_numbers;
        view.pair_shift_rcell.a11 = 1.0f;
        view.pair_shift_rcell.a22 = 2.0f;
        view.pair_shift_rcell.a33 = 3.0f;
        view.pair_shift_bits = pair_shift_bits;
        need_shift.require_gmxpacked_payload = true;
        need_shift.gmxpacked_payload_generation =
            view.gmxpacked_payload_generation;
        need_shift.require_pair_shift_rcell = true;
        need_shift.pair_shift_rcell = view.pair_shift_rcell;
        Expect_Valid("pair-shift capability", view, need_shift);
        {
            auto stale = need_shift;
            stale.gmxpacked_payload_generation += 1;
            Expect_Invalid(
                "stale gmxpacked payload generation",
                "clustered spatial view gmxpacked payload generation is stale",
                view, stale);
        }
        {
            auto stale = view;
            stale.pair_shift_payload_generation += 1;
            Expect_Invalid(
                "stale pair-shift payload generation",
                "clustered spatial view pair-shift payload generation is stale",
                stale, need_shift);
        }
        {
            auto stale = view;
            stale.pair_shift_geometry_generation += 1;
            Expect_Invalid(
                "stale pair-shift geometry generation",
                "clustered spatial view pair-shift geometry generation is stale",
                stale, need_shift);
        }
        {
            auto stale = view;
            stale.pair_shift_cjpacked_numbers += 1;
            Expect_Invalid(
                "stale pair-shift counts",
                "clustered spatial view pair-shift counts are stale", stale,
                need_shift);
        }
        {
            auto stale_requirements = need_shift;
            stale_requirements.pair_shift_rcell.a33 += 1.0f;
            Expect_Invalid(
                "stale pair-shift rcell",
                "clustered spatial view pair-shift rcell does not match",
                view, stale_requirements);
        }
    }
    {
        auto need_grouped = requirements;
        need_grouped.require_native_grouped_sci = true;
        Expect_Invalid(
            "missing native grouped SCI capability",
            "clustered spatial view has no native grouped SCI range",
            fixture.view, need_grouped);
        int grouped_offsets[2] = {0, 1};
        int grouped_ids[1] = {0};
        auto view = fixture.view;
        view.native_grouped_sci_ready = true;
        view.native_grouped_sci_offsets = grouped_offsets;
        view.native_grouped_sci_ids = grouped_ids;
        Expect_Valid("native grouped SCI capability", view, need_grouped);
    }
    {
        auto need_grouped = requirements;
        need_grouped.require_gmxpacked_grouped_sci = true;
        Expect_Invalid(
            "missing gmxpacked grouped SCI capability",
            "clustered spatial view has no gmxpacked grouped SCI range",
            fixture.view, need_grouped);
    }
    {
        CLUSTERED_GMXPACKED_SCI gmxpacked_sci[1] = {};
        CLUSTERED_GMXPACKED_CJ gmxpacked_cjpacked[1] = {};
        CLUSTERED_GMXPACKED_EXCLUSION gmxpacked_exclusions[1] = {};
        int endpoint_offsets[2] = {0, 1};
        CLUSTERED_GMXPACKED_ENDPOINT_REFERENCE endpoint_references[1] = {};
        auto view = fixture.view;
        view.gmxpacked_sci_numbers = 1;
        view.gmxpacked_cjpacked_numbers = 1;
        view.gmxpacked_exclusion_numbers = 1;
        view.gmxpacked_sci = gmxpacked_sci;
        view.gmxpacked_cjpacked = gmxpacked_cjpacked;
        view.gmxpacked_exclusions = gmxpacked_exclusions;
        auto need_endpoint = requirements;
        need_endpoint.require_gmxpacked_endpoint_incidence = true;
        Expect_Invalid(
            "missing endpoint-incidence capability",
            "clustered spatial view has no gmxpacked endpoint incidence",
            view, need_endpoint);

        view.gmxpacked_endpoint_incidence_ready = true;
        view.endpoint_incidence_provider_incarnation =
            view.provider_incarnation;
        view.endpoint_incidence_payload_generation =
            view.gmxpacked_payload_generation;
        view.endpoint_incidence_sci_numbers =
            view.gmxpacked_sci_numbers;
        view.endpoint_incidence_cjpacked_numbers =
            view.gmxpacked_cjpacked_numbers;
        view.endpoint_incidence_super_cluster_numbers =
            view.super_cluster_numbers;
        view.endpoint_incidence_reference_numbers = 1;
        view.endpoint_incidence_offset_tail = 1;
        view.gmxpacked_endpoint_incidence_offsets = endpoint_offsets;
        view.gmxpacked_endpoint_incidence_references =
            endpoint_references;
        Expect_Valid("endpoint-incidence capability", view, need_endpoint);
        {
            auto stale = view;
            stale.endpoint_incidence_provider_incarnation += 1;
            Expect_Invalid(
                "stale endpoint-incidence provider",
                "clustered spatial view endpoint-incidence provider is stale",
                stale, need_endpoint);
        }
        {
            auto stale = view;
            stale.endpoint_incidence_cjpacked_numbers += 1;
            Expect_Invalid(
                "stale endpoint-incidence counts",
                "clustered spatial view endpoint-incidence counts are stale",
                stale, need_endpoint);
        }
        {
            auto oversized = view;
            oversized.endpoint_incidence_reference_numbers =
                2 * kClusteredJGroupSize + 1;
            Expect_Invalid(
                "oversized endpoint-incidence capability",
                "clustered spatial view endpoint-incidence exceeds structural bound",
                oversized, need_endpoint);
        }
        {
            auto stale = view;
            stale.endpoint_incidence_offset_tail = 0;
            Expect_Invalid(
                "stale endpoint-incidence offset tail",
                "clustered spatial view endpoint-incidence offset tail is stale",
                stale, need_endpoint);
        }
        {
            auto stale = view;
            stale.endpoint_incidence_payload_generation += 1;
            Expect_Invalid(
                "stale endpoint-incidence payload",
                "clustered spatial view endpoint-incidence payload is stale",
                stale, need_endpoint);
        }
    }
    {
        CLUSTERED_GMXPACKED_SCI sci[1] = {};
        CLUSTERED_GMXPACKED_CJ cjpacked[1] = {};
        CLUSTERED_GMXPACKED_EXCLUSION exclusions[1] = {};
        auto view = fixture.view;
        view.sci_numbers = 0;
        view.cjpacked_numbers = 0;
        view.exclusion_pool_numbers = 0;
        view.sci = nullptr;
        view.cjpacked = nullptr;
        view.exclusion_mask_pool = nullptr;
        view.gmxpacked_sci_numbers = 1;
        view.gmxpacked_cjpacked_numbers = 1;
        view.gmxpacked_exclusion_numbers = 1;
        view.gmxpacked_sci = sci;
        view.gmxpacked_cjpacked = cjpacked;
        view.gmxpacked_exclusions = exclusions;
        auto gmxpacked_requirements = requirements;
        gmxpacked_requirements.require_native_payload = false;
        gmxpacked_requirements.native_payload_generation = -1;
        gmxpacked_requirements.require_gmxpacked_payload = true;
        gmxpacked_requirements.gmxpacked_payload_generation =
            view.gmxpacked_payload_generation;
        Expect_Valid("gmxpacked-only spatial view", view,
                     gmxpacked_requirements);
    }
}

void Test_Grouped_Sci_Shift_Contract()
{
    int grouped_offsets[2] = {0, 2};
    // Deliberately reverse the SCI order: production grouping is not a
    // shift-sorted contract.
    int grouped_ids[2] = {1, 0};
    CLUSTERED_GMXPACKED_SCI sci[2] = {};
    sci[0].supercluster_id = 0;
    sci[0].shift_id = 12;
    sci[0].cjpacked_begin = 0;
    sci[0].cjpacked_end = 1;
    sci[1].supercluster_id = 0;
    sci[1].shift_id = 14;
    sci[1].cjpacked_begin = 1;
    sci[1].cjpacked_end = 2;

    uint64_t pair_shift_bits[2 * kClusteredJGroupSize] = {};
    Clustered_Set_Pair_Shift_Id(&pair_shift_bits[0], 0, 12);
    Clustered_Set_Pair_Shift_Id(&pair_shift_bits[0], 1, 15);
    Clustered_Set_Pair_Shift_Id(
        &pair_shift_bits[kClusteredJGroupSize], 0, 14);
    Clustered_Set_Pair_Shift_Id(
        &pair_shift_bits[kClusteredJGroupSize], 1, 13);

    CLUSTERED_SPATIAL_VIEW view = {};
    view.gmxpacked_sci_numbers = 2;
    view.gmxpacked_grouped_sci_ready = true;
    view.super_cluster_numbers = 1;
    view.gmxpacked_grouped_sci_offsets = grouped_offsets;
    view.gmxpacked_grouped_sci_ids = grouped_ids;
    view.gmxpacked_sci = sci;
    view.pair_shift_metadata_ready = true;
    view.pair_shift_bits = pair_shift_bits;

    const CLUSTERED_GROUPED_SCI_RANGE range =
        Clustered_Gmxpacked_Grouped_Sci_Range(view, 0);
    Check(range.begin == 0 && range.end == 2,
          "grouped range spans both shift records");
    Check(Clustered_Gmxpacked_Grouped_Sci_Id(view, range.begin) == 1,
          "grouped range preserves structural SCI ordering");
    Check(Clustered_Gmxpacked_Grouped_Sci_Id(view, range.begin + 1) == 0,
          "grouped range contains the second SCI");

    const int first_sci_id =
        Clustered_Gmxpacked_Grouped_Sci_Id(view, range.begin);
    const int second_sci_id =
        Clustered_Gmxpacked_Grouped_Sci_Id(view, range.begin + 1);
    Check(sci[first_sci_id].shift_id == 14 &&
              sci[second_sci_id].shift_id == 12,
          "a grouped range may contain multiple SCI shifts");

    Check(Clustered_Gmxpacked_Pair_Shift_Id(
              view, sci[first_sci_id], sci[first_sci_id].cjpacked_begin, 0,
              0) == 14,
          "first i-cluster uses its explicit pair shift");
    Check(Clustered_Gmxpacked_Pair_Shift_Id(
              view, sci[first_sci_id], sci[first_sci_id].cjpacked_begin, 0,
              1) == 13,
          "one SCI may contain a different pair shift for another i-cluster");
    Check(Clustered_Gmxpacked_Pair_Shift_Id(
              view, sci[second_sci_id], sci[second_sci_id].cjpacked_begin, 0,
              1) == 15,
          "second SCI also resolves its per-pair shift independently");

    // A second traversal starts from the same structural range; no atom-pair
    // adjacency is materialized to make the neighborhood re-iterable.
    const CLUSTERED_GROUPED_SCI_RANGE replay =
        Clustered_Gmxpacked_Grouped_Sci_Range(view, 0);
    Check(replay.begin == range.begin && replay.end == range.end,
          "grouped SCI range is re-iterable");
}

void Test_Gmxpacked_Endpoint_Incidence_Contract()
{
    constexpr int cluster_numbers = 4;
    constexpr int super_cluster_numbers = 3;
    int super_cluster_offsets[super_cluster_numbers + 1] = {0, 2, 3, 4};
    CLUSTERED_GMXPACKED_SCI sci[2] = {};
    sci[0].supercluster_id = 0;
    sci[0].shift_id = 12;
    sci[0].cjpacked_begin = 0;
    sci[0].cjpacked_end = 1;
    sci[1].supercluster_id = 0;
    sci[1].shift_id = 14;
    sci[1].cjpacked_begin = 1;
    sci[1].cjpacked_end = 2;

    CLUSTERED_GMXPACKED_CJ cjpacked[2] = {};
    cjpacked[0].cj[0] = 2;
    cjpacked[0].split[0].imask = (1u << 0) | (1u << 1);
    cjpacked[1].cj[0] = 3;
    cjpacked[1].split[1].imask = (1u << 1);

    CLUSTERED_GMXPACKED_ENDPOINT_INCIDENCE_HOST incidence;
    const char* reason = reinterpret_cast<const char*>(1);
    Check(Clustered_Build_Gmxpacked_Endpoint_Incidence_Host(
              31, 47, cluster_numbers, super_cluster_numbers,
              super_cluster_offsets, 2, sci, 2, cjpacked, &incidence,
              &reason),
          "endpoint-incidence host builder accepts a valid payload");
    Check(reason == nullptr,
          "endpoint-incidence host builder clears failure reason");
    Check(incidence.ready && incidence.provider_incarnation == 31 &&
              incidence.gmxpacked_payload_generation == 47,
          "endpoint incidence pins provider and payload generations");
    Check(incidence.offsets.size() == 4 &&
              incidence.offsets[0] == 0 &&
              incidence.offsets[1] == 2 &&
              incidence.offsets[2] == 3 &&
              incidence.offsets[3] == 4,
          "endpoint incidence groups native and transposed tile endpoints");
    Check(incidence.references.size() == 4,
          "endpoint incidence emits two tile references per valid CJ/jm");

    CLUSTERED_SPATIAL_VIEW view = {};
    view.cluster_numbers = cluster_numbers;
    view.super_cluster_numbers = super_cluster_numbers;
    view.gmxpacked_sci_numbers = 2;
    view.gmxpacked_cjpacked_numbers = 2;
    view.gmxpacked_endpoint_incidence_ready = incidence.ready;
    view.endpoint_incidence_reference_numbers =
        static_cast<int>(incidence.references.size());
    view.gmxpacked_endpoint_incidence_offsets =
        incidence.offsets.data();
    view.gmxpacked_endpoint_incidence_references =
        incidence.references.data();
    view.super_cluster_offsets = super_cluster_offsets;
    view.gmxpacked_sci = sci;
    view.gmxpacked_cjpacked = cjpacked;

    const CLUSTERED_ENDPOINT_INCIDENCE_RANGE native_range =
        Clustered_Gmxpacked_Endpoint_Incidence_Range(view, 0);
    Check(native_range.begin == 0 && native_range.end == 2,
          "native center range spans CJ records from two SCI shifts");
    const auto* first_native =
        Clustered_Gmxpacked_Endpoint_Incidence_Reference(
            view, native_range.begin);
    const auto* second_native =
        Clustered_Gmxpacked_Endpoint_Incidence_Reference(
            view, native_range.begin + 1);
    Check(first_native != nullptr && second_native != nullptr &&
              first_native->orientation ==
                  CLUSTERED_ENDPOINT_ORIENTATION::NATIVE_I &&
              second_native->orientation ==
                  CLUSTERED_ENDPOINT_ORIENTATION::NATIVE_I &&
              sci[first_native->sci_id].shift_id == 12 &&
              sci[second_native->sci_id].shift_id == 14,
          "native center range preserves distinct SCI shift records");
    Check(first_native != nullptr &&
              first_native->i_cluster_mask == 0x3u,
          "multi-cluster imask remains one fixed-width tile reference");

    const CLUSTERED_ENDPOINT_INCIDENCE_RANGE transposed_one =
        Clustered_Gmxpacked_Endpoint_Incidence_Range(view, 1);
    const auto* first_transposed =
        Clustered_Gmxpacked_Endpoint_Incidence_Reference(
            view, transposed_one.begin);
    Check(transposed_one.begin == 2 && transposed_one.end == 3 &&
              first_transposed != nullptr &&
              first_transposed->orientation ==
                  CLUSTERED_ENDPOINT_ORIENTATION::TRANSPOSED_J &&
              cjpacked[first_transposed->cjpacked_id]
                      .cj[first_transposed->jm] == 2,
          "a J endpoint can replay a half-payload tile in transposed orientation");

    CLUSTERED_GMXPACKED_CENTER_CURSOR native_cursor;
    Check(Clustered_Gmxpacked_Center_Cursor_Begin(
              view, 1, &native_cursor),
          "center cursor begins for a native I cluster");
    CLUSTERED_GMXPACKED_CENTER_TILE native_tile = {};
    Check(Clustered_Gmxpacked_Center_Cursor_Next(
              view, &native_cursor, &native_tile) &&
              native_tile.orientation ==
                  CLUSTERED_ENDPOINT_ORIENTATION::NATIVE_I &&
              native_tile.center_cluster == 1 &&
              native_tile.neighbor_cluster_base == 2 &&
              native_tile.neighbor_cluster_mask == 1u &&
              Clustered_Gmxpacked_Center_Tile_Pair_Shift_Id(
                  view, native_tile, 0) == 12,
          "native center cursor yields one J cluster with the original shift");
    Check(Clustered_Gmxpacked_Center_Cursor_Next(
              view, &native_cursor, &native_tile) &&
              native_tile.neighbor_cluster_base == 3 &&
              Clustered_Gmxpacked_Center_Tile_Pair_Shift_Id(
                  view, native_tile, 0) == 14,
          "native center cursor replays records from a second SCI shift");
    Check(!Clustered_Gmxpacked_Center_Cursor_Next(
              view, &native_cursor, &native_tile),
          "native center cursor terminates after its structural range");

    CLUSTERED_GMXPACKED_CENTER_CURSOR transposed_cursor;
    Check(Clustered_Gmxpacked_Center_Cursor_Begin(
              view, 2, &transposed_cursor),
          "center cursor begins for a transposed J cluster");
    CLUSTERED_GMXPACKED_CENTER_TILE transposed_tile = {};
    Check(Clustered_Gmxpacked_Center_Cursor_Next(
              view, &transposed_cursor, &transposed_tile) &&
              transposed_tile.orientation ==
                  CLUSTERED_ENDPOINT_ORIENTATION::TRANSPOSED_J &&
              transposed_tile.neighbor_cluster_base == 0 &&
              transposed_tile.neighbor_cluster_mask == 0x3u &&
              Clustered_Gmxpacked_Center_Tile_Pair_Shift_Id(
                  view, transposed_tile, 0) == 14 &&
              Clustered_Gmxpacked_Center_Tile_Pair_Shift_Id(
                  view, transposed_tile, 1) == 14,
          "transposed cursor yields the original I mask and reverses its shift");
    Check(!Clustered_Gmxpacked_Center_Cursor_Next(
              view, &transposed_cursor, &transposed_tile),
          "transposed center cursor terminates after its structural range");

    CLUSTERED_GMXPACKED_CENTER_CURSOR partial_cursor;
    Check(Clustered_Gmxpacked_Center_Cursor_Begin(
              view, 3, &partial_cursor) &&
              Clustered_Gmxpacked_Center_Cursor_Next(
                  view, &partial_cursor, &transposed_tile) &&
              transposed_tile.neighbor_cluster_mask == 0x2u &&
              Clustered_Gmxpacked_Center_Tile_Pair_Shift_Id(
                  view, transposed_tile, 0) == -1 &&
              Clustered_Gmxpacked_Center_Tile_Pair_Shift_Id(
                  view, transposed_tile, 1) == 12,
          "transposed cursor preserves a partial source-cluster mask");
    Check(Clustered_Invert_Shift_Id(0) == 26 &&
              Clustered_Invert_Shift_Id(13) == 13 &&
              Clustered_Invert_Shift_Id(26) == 0 &&
              Clustered_Invert_Shift_Id(-1) == -1,
          "clustered shift inversion is total over the 27-image domain");

    const CLUSTERED_ENDPOINT_INCIDENCE_RANGE replay =
        Clustered_Gmxpacked_Endpoint_Incidence_Range(view, 0);
    Check(replay.begin == native_range.begin &&
              replay.end == native_range.end,
          "two endpoint cursors can replay one deterministic range");

    CLUSTERED_GMXPACKED_ENDPOINT_INCIDENCE_HOST rebuilt;
    Check(Clustered_Build_Gmxpacked_Endpoint_Incidence_Host(
              31, 47, cluster_numbers, super_cluster_numbers,
              super_cluster_offsets, 2, sci, 2, cjpacked, &rebuilt,
              nullptr),
          "endpoint incidence can be rebuilt for the same generation");
    Check(rebuilt.offsets == incidence.offsets &&
              rebuilt.references.size() == incidence.references.size() &&
              std::memcmp(rebuilt.references.data(),
                          incidence.references.data(),
                          sizeof(CLUSTERED_GMXPACKED_ENDPOINT_REFERENCE) *
                              incidence.references.size()) == 0,
          "endpoint-incidence ordering is deterministic");

    auto reset_view = view;
    reset_view.gmxpacked_endpoint_incidence_ready = false;
    Check(Clustered_Gmxpacked_Endpoint_Incidence_Range(reset_view, 0).end == 0,
          "reset endpoint-incidence state rejects stale cursor access");

    int malformed_range_offsets[super_cluster_numbers + 1] = {0, 5, 3, 4};
    auto malformed_range_view = view;
    malformed_range_view.gmxpacked_endpoint_incidence_offsets =
        malformed_range_offsets;
    const CLUSTERED_ENDPOINT_INCIDENCE_RANGE malformed_range =
        Clustered_Gmxpacked_Endpoint_Incidence_Range(
            malformed_range_view, 0);
    Check(malformed_range.begin == 0 && malformed_range.end == 0,
          "endpoint-incidence cursor rejects ranges beyond reference storage");

    int invalid_offsets[super_cluster_numbers + 1] = {0, 2, 3, 3};
    CLUSTERED_GMXPACKED_ENDPOINT_INCIDENCE_HOST invalid;
    reason = nullptr;
    Check(!Clustered_Build_Gmxpacked_Endpoint_Incidence_Host(
              31, 47, cluster_numbers, super_cluster_numbers,
              invalid_offsets, 2, sci, 2, cjpacked, &invalid, &reason) &&
              !invalid.ready && reason != nullptr &&
              std::strcmp(
                  reason,
                  "clustered endpoint-incidence supercluster offsets are invalid") ==
                  0,
          "invalid grouping fails closed and clears endpoint state");
}

void Test_Pair_Shift_And_Exclusion_Decoding()
{
    CLUSTERED_GMXPACKED_SCI sci = {};
    sci.shift_id = kClusteredCentralShiftId;
    sci.cjpacked_begin = 0;
    sci.cjpacked_end = 1;

    CLUSTERED_GMXPACKED_CJ packed = {};
    packed.cj[0] = 0;
    packed.split[0].imask = (1u << 0) | (1u << 1);
    packed.split[0].exclusion_index = 1;

    CLUSTERED_GMXPACKED_EXCLUSION exclusions[2] = {};
    for (int lane_pair = 0;
         lane_pair < kClusteredGmxpackedExclusionPairCount; lane_pair += 1)
    {
        exclusions[1].pair[lane_pair] = 0xffffffffu;
    }
    // For i_lane=3 and j_lane=2, exclude jm=0/i_local=0 while retaining
    // jm=0/i_local=1.
    const int i_lane = 3;
    const int j_lane = 2;
    const int split = j_lane / kClusteredSplitJClusterSize;
    const int split_j_lane =
        j_lane - split * kClusteredSplitJClusterSize;
    const int pair_word =
        split_j_lane * kClusteredClusterSize + i_lane;
    exclusions[1].pair[pair_word] &= ~(1u << 0);

    uint64_t pair_shift_bits[kClusteredJGroupSize] = {};
    Clustered_Set_Pair_Shift_Id(&pair_shift_bits[0], 0, 22);
    Clustered_Set_Pair_Shift_Id(&pair_shift_bits[0], 1, 16);

    CLUSTERED_SPATIAL_VIEW view = {};
    view.pair_shift_metadata_ready = true;
    view.pair_shift_bits = pair_shift_bits;

    const unsigned int pair_bits = exclusions[1].pair[pair_word];
    const unsigned int effective_mask =
        packed.split[split].imask & pair_bits;
    Check((effective_mask & (1u << 0)) == 0u,
          "gmxpacked exclusion removes the selected i-cluster pair");
    Check((effective_mask & (1u << 1)) != 0u,
          "gmxpacked exclusion preserves an unrelated i-cluster pair");
    Check(Clustered_Gmxpacked_Pair_Shift_Id(view, sci, 0, 0, 0) == 22,
          "gmxpacked pair shift decodes a non-central image");
    Check(Clustered_Gmxpacked_Pair_Shift_Id(view, sci, 0, 0, 1) == 16,
          "gmxpacked pair shift remains lane-specific");

    LTMatrix3 cell = {};
    cell.a11 = 10.0f;
    cell.a22 = 20.0f;
    cell.a33 = 30.0f;
    const VECTOR x_shift = Clustered_Shift_Vector_From_Id(22, cell);
    Check(x_shift.x == 10.0f && x_shift.y == 0.0f &&
              x_shift.z == 0.0f,
          "non-central pair shift maps through the current cell");

    CLUSTERED_CJ_PACKED native = {};
    for (int native_split = 0; native_split < kClusteredWarpSplitCount;
         native_split += 1)
    {
        for (int index = 0;
             index < kClusteredJGroupSize *
                         kClusteredSuperClusterClusters;
             index += 1)
        {
            native.imei[native_split].excl_ind[index] = -1;
        }
    }
    native.cj[0] = 0;
    native.imei[0].imask = 1u;
    Clustered_Exclusion_Index_Ref(native.imei[0], 0, 0) = 0;
    const uint64_t native_exclusion =
        1ull << static_cast<unsigned int>(i_lane * kClusteredClusterSize +
                                          j_lane);
    Check(Clustered_First_Exclusion_Index(native, 0, 0) == 0,
          "native payload resolves the explicit exclusion mask");
    Check((native_exclusion &
           (1ull << static_cast<unsigned int>(
                i_lane * kClusteredClusterSize + j_lane))) != 0ull,
          "native exclusion uses the same i-lane/j-lane identity");
    Check(Clustered_First_Exclusion_Index(native, 0, 1) == -1,
          "native payload leaves unrelated i-cluster pairs unexcluded");
}

void Test_Local_Ghost_Pair_Ownership()
{
    Check(Clustered_Local_I_Owns_Pair(5, 7, 2, 0, false),
          "a ghost J lane survives reverse cluster ordering");
    Check(Clustered_Local_I_Owns_Pair(2, 7, 5, 0, true),
          "forward local-local cluster ordering is owned");
    Check(!Clustered_Local_I_Owns_Pair(5, 0, 2, 7, true),
          "reverse local-local cluster ordering is rejected");
    Check(Clustered_Local_I_Owns_Pair(3, 1, 3, 2, true),
          "forward lane ordering in one local cluster is owned");
    Check(!Clustered_Local_I_Owns_Pair(3, 2, 3, 1, true),
          "reverse lane ordering in one local cluster is rejected");
    Check(!Clustered_Local_I_Owns_Pair(3, 2, 3, 2, true),
          "a local lane does not pair with itself");
    Check(Clustered_Local_I_Owns_Pair(3, 2, 3, 1, false),
          "a ghost lane in a mixed cluster survives reverse lane ordering");
}

void Test_Pair_Shift_Cache_Lifecycle()
{
    CLUSTERED_GMXPACKED_PAIR_SHIFT_CACHE_KEY current = {};
    current.gmxpacked_payload_generation = 7;
    current.geometry_generation = 13;
    current.sci_numbers = 3;
    current.cjpacked_numbers = 5;
    current.exclusion_numbers = 2;
    current.rcell.a11 = 1.0f;
    current.rcell.a21 = 2.0f;
    current.rcell.a22 = 3.0f;
    current.rcell.a31 = 4.0f;
    current.rcell.a32 = 5.0f;
    current.rcell.a33 = 6.0f;
    const CLUSTERED_GMXPACKED_PAIR_SHIFT_CACHE_KEY cached = current;

    Check(!Clustered_Gmxpacked_Pair_Shift_Metadata_Should_Refresh(
              true, true, true, cached, current),
          "identical payload/geometry generations, counts and rcell reuse pair shifts");

    {
        auto changed = current;
        changed.gmxpacked_payload_generation += 1;
        Check(Clustered_Gmxpacked_Pair_Shift_Metadata_Should_Refresh(
                  true, true, true, cached, changed),
              "same-count payload replacement refreshes pair shifts");
    }
    {
        auto changed = current;
        changed.geometry_generation += 1;
        Check(Clustered_Gmxpacked_Pair_Shift_Metadata_Should_Refresh(
                  true, true, true, cached, changed),
              "same-count coordinate gather refreshes pair shifts");
    }
    {
        auto changed = current;
        changed.sci_numbers += 1;
        Check(Clustered_Gmxpacked_Pair_Shift_Metadata_Should_Refresh(
                  true, true, true, cached, changed),
              "SCI count change refreshes pair shifts");
    }
    {
        auto changed = current;
        changed.cjpacked_numbers += 1;
        Check(Clustered_Gmxpacked_Pair_Shift_Metadata_Should_Refresh(
                  true, true, true, cached, changed),
              "CJ count change refreshes pair shifts");
    }
    {
        auto changed = current;
        changed.exclusion_numbers += 1;
        Check(Clustered_Gmxpacked_Pair_Shift_Metadata_Should_Refresh(
                  true, true, true, cached, changed),
              "exclusion count change refreshes pair shifts");
    }
    {
        auto changed = current;
        changed.rcell.a32 += 1.0f;
        Check(Clustered_Gmxpacked_Pair_Shift_Metadata_Should_Refresh(
                  true, true, true, cached, changed),
              "rcell change refreshes pair shifts independently of payload");
    }
    Check(Clustered_Gmxpacked_Pair_Shift_Metadata_Should_Refresh(
              true, false, true, cached, current),
          "missing pair-shift storage refreshes metadata");
    Check(Clustered_Gmxpacked_Pair_Shift_Metadata_Should_Refresh(
              true, true, false, cached, current),
          "reset metadata state refreshes pair shifts");
    Check(Clustered_Gmxpacked_Pair_Shift_Metadata_Should_Refresh(
              false, true, true, cached, current),
          "disabled cache always refreshes pair shifts");
}

void Test_View_Factory_Rejects_Unpublished_Cache()
{
    LJ_CLUSTERED_DIRECT_CACHE cache = {};
    cache.initialized = true;
    cache.layout.enabled = true;
    cache.layout.cache_ready = false;

    CLUSTERED_SPATIAL_VIEW view = {};
    const char* reason = nullptr;
    Check(!Make_Clustered_Spatial_View_From_LJ_Cache(&cache, &view, &reason),
          "failed/domain-invalidated build cannot publish a view");
    Check(reason != nullptr &&
              std::strcmp(reason,
                          "clustered payload cache is disabled or stale") == 0,
          "unpublished cache reports an exact lifecycle failure");
}

void Test_Clear_Retires_Provider_Lifecycle()
{
    LJ_CLUSTER_LAYOUT layout = {};
    layout.provider_incarnation = 41;
    layout.spatial_view_lease_epoch = 50;
    layout.native_payload_generation = 60;
    layout.gmxpacked_compact_payload_generation = 70;
    layout.geometry_generation = 80;

    layout.Retire_Spatial_Provider_Lifetime();

    Check(layout.provider_incarnation == 42,
          "Clear advances provider incarnation without resetting it");
    Check(layout.spatial_view_lease_epoch == 51,
          "Clear retires all borrowed view leases");
    Check(layout.native_payload_generation == 61,
          "Clear advances native payload generation");
    Check(layout.gmxpacked_compact_payload_generation == 71,
          "Clear advances gmxpacked payload generation");
    Check(layout.geometry_generation == 81,
          "Clear advances geometry generation");
}

}  // namespace

int main()
{
    Test_Validation_Contract();
    Test_Grouped_Sci_Shift_Contract();
    Test_Gmxpacked_Endpoint_Incidence_Contract();
    Test_Pair_Shift_And_Exclusion_Decoding();
    Test_Local_Ghost_Pair_Ownership();
    Test_Pair_Shift_Cache_Lifecycle();
    Test_View_Factory_Rejects_Unpublished_Cache();
    Test_Clear_Retires_Provider_Lifecycle();
    if (failures != 0)
    {
        std::fprintf(stderr, "%d clustered spatial-view checks failed\n",
                     failures);
        return 1;
    }
    std::printf("clustered spatial-view contract checks passed\n");
    return 0;
}
