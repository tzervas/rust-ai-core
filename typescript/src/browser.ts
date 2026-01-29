/**
 * Browser WebAssembly bindings for @rust-ai/core
 * @module
 */

import type { DType, DeviceInfo, LogLevel, MemoryTrackerInterface } from './types.js';

interface WasmBindings {
  MemoryTrackerWasm: { create: (limitGb?: number, overheadFactor?: number) => WasmMemoryTracker };
  estimateTensorBytesWasm: (shape: Uint32Array, dtype?: string) => number;
  estimateAttentionMemoryWasm: (batchSize: number, numHeads: number, seqLen: number, headDim: number, dtype?: string) => number;
  cudaAvailableWasm: () => boolean;
  getDeviceInfoWasm: () => DeviceInfo;
  bytesPerDtypeWasm: (dtype: string) => number;
  isFloatingPointDtypeWasm: (dtype: string) => boolean;
  accumulatorDtypeWasm: (dtype: string) => string;
  supportedDtypesWasm: () => string[];
  initLoggingWasm: (level?: string) => void;
  versionWasm: () => string;
  defaultOverheadFactorWasm: () => number;
}

interface WasmMemoryTracker {
  wouldFit(bytes: number): boolean;
  allocate(bytes: number): void;
  deallocate(bytes: number): void;
  allocatedBytes(): number;
  peakBytes(): number;
  limitBytes(): number;
  estimateWithOverhead(shape: Uint32Array, dtype?: string): number;
  reset(): void;
}

let wasm: WasmBindings | null = null;

export async function init(wasmModule?: WasmBindings): Promise<void> {
  if (wasm !== null) return;
  if (wasmModule) {
    wasm = wasmModule;
  } else {
    const module = await import('./wasm/rust_ai_core.js') as { default: () => Promise<WasmBindings> };
    wasm = await module.default();
  }
}

function ensureInitialized(): WasmBindings {
  if (wasm === null) throw new Error('@rust-ai/core: WASM module not initialized. Call `await init()` first.');
  return wasm;
}

export class MemoryTracker implements MemoryTrackerInterface {
  private readonly inner: WasmMemoryTracker;

  constructor(limitGb = 2.0, overheadFactor?: number) {
    const w = ensureInitialized();
    this.inner = w.MemoryTrackerWasm.create(limitGb, overheadFactor);
  }

  wouldFit(bytes: number): boolean { return this.inner.wouldFit(bytes); }
  allocate(bytes: number): void { this.inner.allocate(bytes); }
  deallocate(bytes: number): void { this.inner.deallocate(bytes); }
  allocatedBytes(): number { return this.inner.allocatedBytes(); }
  peakBytes(): number { return this.inner.peakBytes(); }
  limitBytes(): number { return this.inner.limitBytes(); }
  estimateWithOverhead(shape: number[], dtype: DType = 'f32'): number {
    return this.inner.estimateWithOverhead(new Uint32Array(shape), dtype);
  }
  reset(): void { this.inner.reset(); }
}

export function estimateTensorBytes(shape: number[], dtype: DType = 'f32'): number {
  return ensureInitialized().estimateTensorBytesWasm(new Uint32Array(shape), dtype);
}

export function estimateAttentionMemory(batchSize: number, numHeads: number, seqLen: number, headDim: number, dtype: DType = 'bf16'): number {
  return ensureInitialized().estimateAttentionMemoryWasm(batchSize, numHeads, seqLen, headDim, dtype);
}

export function cudaAvailable(): boolean { return ensureInitialized().cudaAvailableWasm(); }
export function getDeviceInfo(): DeviceInfo { return ensureInitialized().getDeviceInfoWasm(); }
export function bytesPerDtype(dtype: DType): number { return ensureInitialized().bytesPerDtypeWasm(dtype); }
export function isFloatingPointDtype(dtype: DType): boolean { return ensureInitialized().isFloatingPointDtypeWasm(dtype); }
export function accumulatorDtype(dtype: DType): DType { return ensureInitialized().accumulatorDtypeWasm(dtype) as DType; }
export function supportedDtypes(): DType[] { return ensureInitialized().supportedDtypesWasm() as DType[]; }
export function initLogging(level: LogLevel = 'info'): void { ensureInitialized().initLoggingWasm(level); }
export function version(): string { return ensureInitialized().versionWasm(); }
export function defaultOverheadFactor(): number { return ensureInitialized().defaultOverheadFactorWasm(); }

export type { DType, DeviceInfo, LogLevel, MemoryTrackerInterface } from './types.js';
export default init;
