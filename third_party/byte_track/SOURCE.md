# ByteTrack source snapshot

This directory contains the tracker core adapted from the official ByteTrack
repository:

- Source: https://github.com/FoundationVision/ByteTrack
- Upstream commit used for the snapshot: `d1bf0191adff59bc8fcfeaa0b33d3d1642552a99`
- License: MIT (see `LICENSE`)

Only the tracker modules needed by the adapter are included. The snapshot has
small compatibility edits for the NumPy 2.x `np.float` removal, removal of an
unused optional torch import, and a minimal package initializer; these edits
are noted here. The default configuration requests the official adapter; if
SciPy/LAP/Cython dependencies are not available, the runtime falls back to
`simple_bytetrack` automatically.
