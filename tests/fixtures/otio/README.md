# OTIO corpus provenance

These are unmodified repository-owned sample documents, not issue attachments.
Source: AcademySoftwareFoundation/OpenTimelineIO, immutable commit
[`bc5fe2d78dc3f8b2a8feb7e04483d85a12e80072`](https://github.com/AcademySoftwareFoundation/OpenTimelineIO/commit/bc5fe2d78dc3f8b2a8feb7e04483d85a12e80072),
directory `tests/sample_data/`. License and upstream NOTICE accompany the files.

| File | Bytes | SHA256 |
| --- | ---: | --- |
| effects.otio | 356446 | 2507d49a7c14b391116da515521bd087dee238fbb27b9a83faec0dc1ffeb151c |
| nested_example.otio | 13242 | 2d11b95eb34522952b8b3d99ddac8935be6700649fd9726f6f2424bacd22ccd0 |
| clip_example.otio | 3750 | b87fbdaf96561c4ba2b361c572b298a91aa2a73613e215402dd5ae400220b326 |
| multiple_track.otio | 13052 | 8cfcaa5f346a302a3e3c23ed03b35f85046c9d9a6a67e5c894652d814a0b5668 |

`effects.otio` carries Resolve_OTIO metadata and has eight video tracks, eleven
audio tracks, 37 clips, 23 gaps, and one dissolve. Its introducing commit,
[`1b3ece18`](https://github.com/AcademySoftwareFoundation/OpenTimelineIO/commit/1b3ece18f928fcf542f24be0bd9a730832002dc4),
adds the otiotool remove-effects test. Neither that commit nor PR #1912
documents an editor export procedure or source project. Resolve metadata alone
does not prove export provenance; this is not a verified Resolve export.
Its media files are not included; the tests check structural equivalence,
not rendering of its foreign URLs. The other samples exercise nested
compositions/time effects, dissolve placement, and multiple tracks.

The tests deserialize with OpenTimelineIO 0.18.1, import/export through Dinkster,
and call the actual OTIO library's `is_equivalent_to`. The CPU memory workload
in `tools/timeline_conformance.py` is separately labeled synthetic. It must
not be presented as a Resolve or Kdenlive export.
