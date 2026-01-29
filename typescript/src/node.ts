/**
 * Node.js native bindings for @rust-ai/core
 * @module
 */

// eslint-disable-next-line @typescript-eslint/no-require-imports
const native = require('../rust-ai-core.node') as NativeBindings;

import type { DType, DeviceInfo, LogLevel, MemoryTrackerInterface } from './types.js';

interface NativeBindings {
  MemoryTracker: new (limitGb?: number, overheadFactor?: number) => NativeMemoryTracker;
  estimateTensorBytes: (shape: number[], dtype?: string) => number;
  estimateAttentionMemory: (batchSize: number, numHeads: number, seqLen: number, headDim: number, dtype?: string) => number;
  cudaAvailable: () => boolean;
  getDeviceInfo: (forceCpu?: boolean, cudaDevice?: number) => DeviceInfo;
  bytesPerDtype: (dtype: string) => number;
  isFloatingPointDtype: (dtype: string) => boolean;
  accumulatorDtype: (dtype: string) => string;
  supportedDtypes: () => string[];
  initLogging: (level?: string, timestamps?: boolean, ansi?: boolean) => void;
  version: () => string;
  defaultOverheadFactor: () => number;
}

interface NativeMemoryTracker {
  wouldFit(bytes: number): boolean;
  allocate(bytes: number): void;
  deallocate(bytes: number): void;
  allocatedBytes(): number;
  peakBytes(): number;
  limitBytes(): number;
  estimateWithOverhead(shape: number[], dtype?: string): number;
  reset(): void;
}

export class MemoryTracker implements MemoryTrackerInterface {
  private readonly inner: NativeMemoryTracker;

  constructor(limitGb = 8.0, overheadFactor?: number) {
    this.inner = new native.MemoryTracker(limitGb, overheadFactor);
  }

  wouldFit(bytes: number): boolean { return this.inner.wouldFit(bytes); }
  allocate(bytes: number): void { this.inner.allocate(bytes); }
  deallocate(bytes: number): void { this.inner.deallocate(bytes); }
  allocatedBytes(): number { return this.inner.allocatedBytes(); }
  peakBytes(): number { return this.inner.peakBytes(); }
  limitBytes(): number { return this.inner.limitBytes(); }
  estimateWithOverhead(shape: number[], dtype: DType = 'f32'): number {
    return this.inner.estimateWithOverhead(shape, dtype);
  }
  reset(): void { this.inner.reset(); }
}

export function estimateTensorBytes(shape: number[], dtype: DType = 'f32'): number {
  return native.estimateTensorBytes(shape, dtype);
}

export function estimateAttentionMemory(batchSize: number, numHeads: number, seqLen: number, headDim: number, dtype: DType = 'bf16'): number {
  return native.estimateAttentionMemory(batchSize, numHeads, seqLen, headDim, dtype);
}

export function cudaAvailable(): boolean { return native.cudaAvailable(); }
export function getDeviceInfo(forceCpu = false, cudaDevice = 0): DeviceInfo { return native.getDeviceInfo(forceCpu, cudaDevice); }
export function bytesPerDtype(dtype: DType): number { return native.bytesPerDtype(dtype); }
export function isFloatingPointDtype(dtype: DType): boolean { return native.isFloatingPointDtype(dtype); }
export function accumulatorDtype(dtype: DType): DType { return native.accumulatorDtype(dtype) as DType; }
export function supportedDtypes(): DType[] { return native.supportedDtypes() as DType[]; }
export function initLogging(level: LogLevel = 'info', timestamps = true, ansi = true): void { native.initLogging(level, timestamps, ansi); }
export function version(): string { return native.version(); }
export function defaultOverheadFactor(): number { return native.defaultOverheadFactor(); }

export type { DType, DeviceInfo, LogLevel, MemoryTrackerInterface } from './types.js';
