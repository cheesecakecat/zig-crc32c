# zig-crc32c

[![Zig](https://img.shields.io/badge/Zig-0.13.0-orange.svg)](https://ziglang.org)
[![License](https://img.shields.io/badge/license-MIT-orange.svg)](LICENSE)

A CRC32C (Castagnoli) implementation in Zig featuring hardware acceleration and Triple Modular Redundancy for fault tolerance. 

## Implementation Details

- Hardware acceleration via SSE4.2 (x86_64) and CRC32 (ARM64) instructions
- Software fallback using optimized table-based calculation
- Triple redundancy with majority voting for fault mitigation
- Compile-time verification of lookup tables
- Stack-only memory allocation model
- Maximum input size: 1MB per calculation

## Performance

Measured on x86_64 with SSE4.2 (Debug build):

| Implementation | Buffer Size | Cycles/Byte |
|---------------|-------------|-------------|
| Hardware      | 4KB         | ~4          |
| Hardware      | 256B        | ~8          |
| Software      | 4KB         | ~35         |
| Software      | 256B        | ~45         |

## Verification

```bash
zig build test
```

## License

MIT - See [LICENSE](LICENSE)
