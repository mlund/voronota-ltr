// C++ voronota-lt reference harness for the contact `central` flag.
//
// Build & run to regenerate tests/data/central_golden.tsv:
//   c++ -std=c++17 -O2 -I <voronota>/expansion_lt/src central_golden.cpp -o cg && ./cg
// where <voronota> is a checkout of github.com/kliment-olechnovic/voronota (MIT).
// The fixture below MUST match tests/centrality_golden.rs.

#include "voronotalt/voronotalt.h"
#include <cstdio>
#include <vector>
int main() {
    const double pr = 1.4, r = 1.5;          // probe baked into the radius
    const double pts[][3] = {
        {0,0,0},{2.5,0,0},{1.25,2.0,0},{1.25,0.8,2.0},
        {0,2.5,1.0},{2.5,2.5,1.0},{1.25,-1.5,1.0},{3.5,1.25,1.0}
    };
    std::vector<voronotalt::SimpleSphere> spheres;
    for (const auto& p : pts) spheres.emplace_back(p[0], p[1], p[2], r + pr);
    voronotalt::RadicalTessellation::Result result;
    voronotalt::RadicalTessellation::construct_full_tessellation(spheres, result);
    for (const auto& c : result.contacts_summaries)
        // Mask the centrality bit (bit 0) out of `flags`, so the golden stays a
        // boolean even if voronota-lt later packs more bits into the field.
        printf("%u\t%u\t%u\n", (unsigned)c.id_a, (unsigned)c.id_b, (unsigned)(c.flags & 1u));
    return 0;
}
