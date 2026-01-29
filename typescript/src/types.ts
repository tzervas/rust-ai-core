/**
 * TypeScript type definitions for @rust-ai/core
 */

export type DType = 'f16' | 'bf16' | 'f32' | 'f64' | 'u8' | 'u32' | 'i16' | 'i32' | 'i64';
export type DeviceType = 'cuda' | 'metal' | 'cpu';
export type LogLevel = 'trace' | 'debug' | 'info' | 'warn' | 'error';

export interface DeviceInfo {
  type: DeviceType;
  ordinal: number | null;
  name: string;
}

export interface MemoryTrackerInterface {
  wouldFit(bytes: number): boolean;
  allocate(bytes: number): void;
  deallocate(bytes: number): void;
  allocatedBytes(): number;
  peakBytes(): number;
  limitBytes(): number;
  estimateWithOverhead(shape: number[], dtype?: DType): number;
  reset(): void;
}
