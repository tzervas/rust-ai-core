# @rust-ai/core

TypeScript/JavaScript bindings for [rust-ai-core](https://github.com/tzervas/rust-ai-core).

## Installation

```bash
npm install @rust-ai/core
```

## Quick Start

### Node.js

```typescript
import { estimateTensorBytes, MemoryTracker, cudaAvailable, getDeviceInfo } from '@rust-ai/core';

if (cudaAvailable()) {
  console.log(`Using ${getDeviceInfo().name}`);
}

const bytes = estimateTensorBytes([32, 1024, 768], 'bf16');
const tracker = new MemoryTracker(8.0);
tracker.allocate(bytes);
console.log(`Peak: ${tracker.peakBytes()} bytes`);
```

### Browser (WASM)

```typescript
import init, { estimateTensorBytes, MemoryTracker } from '@rust-ai/core/wasm';

await init();
const bytes = estimateTensorBytes([32, 512, 768], 'f32');
```

## API

- `estimateTensorBytes(shape, dtype?)` - Memory estimation
- `estimateAttentionMemory(batch, heads, seq, dim, dtype?)` - Attention memory
- `MemoryTracker` - Track allocations with limits
- `cudaAvailable()` - CUDA check
- `getDeviceInfo()` - Device detection
- `bytesPerDtype(dtype)` - Element size
- `supportedDtypes()` - List dtypes
- `version()` - Get version

## License

MIT
