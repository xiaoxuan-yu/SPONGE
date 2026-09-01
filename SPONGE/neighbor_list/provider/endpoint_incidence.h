#pragma once

#include "state.h"

class ClusteredNeighborProvider;

// Internal endpoint-incidence lifecycle and builder services. These are kept
// out of the public clustered spatial-view API.
void Invalidate_Gmxpacked_Endpoint_Incidence(ClusteredNeighborProvider* layout);
void Build_Gmxpacked_Endpoint_Incidence_Metadata(ClusteredNeighborProvider* layout);
