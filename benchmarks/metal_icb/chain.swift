// A chain of small dependent compute kernels, the shape of one LLM decode token
// on MLX (~1,700 kernels, each waiting on the previous one), submitted four ways:
//
//   serial   one command buffer, one serial compute encoder (Metal orders the
//            dispatches itself)
//   mlx      what MLX does: a concurrent encoder with a buffer memory barrier
//            between dependent dispatches, a new command buffer every 400 ops
//   icb      an indirect command buffer encoded ONCE (each command with a
//            barrier), then replayed with executeCommandsInBuffer — Metal's
//            counterpart to a CUDA graph
//   one      a single dispatch doing the same total work (the floor)
//
// Two workloads: "tiny" (each kernel touches a 10,240-float vector, pure
// dispatch cost) and "weights" (each kernel also streams a 2 MiB weight slab,
// ~3.4 GB per chain, like a decode token's weights).
//
//   swiftc -O chain.swift -o chain && ./chain [kernels] [reps]
import Foundation
import Metal

let K = CommandLine.arguments.count > 1 ? Int(CommandLine.arguments[1])! : 1700
let reps = CommandLine.arguments.count > 2 ? Int(CommandLine.arguments[2])! : 20
let n: UInt32 = 10240            // floats per activation vector
let slabBytes = 2 << 20          // weight bytes per kernel in the "weights" workload
let slabs = 64                   // 128 MiB ring: larger than the 48 MB system cache
let per = UInt32(slabBytes / 16) / n   // float4 reads per thread

let dev = MTLCreateSystemDefaultDevice()!
let queue = dev.makeCommandQueue()!
let src = """
#include <metal_stdlib>
using namespace metal;
kernel void tiny(device const float* a [[buffer(0)]], device float* b [[buffer(1)]],
                 constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < n) b[i] = a[i] * 0.999f + 0.001f;
}
kernel void weights(device const float* a [[buffer(0)]], device float* b [[buffer(1)]],
                    device const float4* w [[buffer(2)]], constant uint& n [[buffer(3)]],
                    constant uint& per [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    float4 acc = 0;
    for (uint j = 0; j < per; j++) acc += w[j * n + i];   // coalesced across threads
    b[i] = a[i] * 0.999f + (acc.x + acc.y + acc.z + acc.w) * 1e-9f;
}
kernel void stream(device const float4* w [[buffer(0)]], device float* out [[buffer(1)]],
                   constant uint& count [[buffer(2)]], uint i [[thread_position_in_grid]],
                   uint g [[threads_per_grid]]) {
    float4 acc = 0;
    for (uint j = i; j < count; j += g) acc += w[j];
    out[i] = acc.x + acc.y + acc.z + acc.w;
}
"""
let lib = try! dev.makeLibrary(source: src, options: nil)
func pipeline(_ name: String) -> MTLComputePipelineState {
    let d = MTLComputePipelineDescriptor()
    d.computeFunction = lib.makeFunction(name: name)!
    d.supportIndirectCommandBuffers = true
    return try! dev.makeComputePipelineState(descriptor: d, options: [], reflection: nil)
}
let pTiny = pipeline("tiny"), pW = pipeline("weights"), pStream = pipeline("stream")

let bufA = dev.makeBuffer(length: Int(n) * 4, options: .storageModeShared)!
let bufB = dev.makeBuffer(length: Int(n) * 4, options: .storageModeShared)!
let wRing = dev.makeBuffer(length: slabBytes * slabs, options: .storageModePrivate)!
var nn = n, pp = per
let bufN = dev.makeBuffer(bytes: &nn, length: 4, options: .storageModeShared)!
let bufPer = dev.makeBuffer(bytes: &pp, length: 4, options: .storageModeShared)!
let tg = MTLSize(width: 256, height: 1, depth: 1)
let grid = MTLSize(width: (Int(n) + 255) / 256, height: 1, depth: 1)

func bind(_ enc: MTLComputeCommandEncoder, _ k: Int, _ withW: Bool) {
    let (inp, out) = k % 2 == 0 ? (bufA, bufB) : (bufB, bufA)
    enc.setBuffer(inp, offset: 0, index: 0)
    enc.setBuffer(out, offset: 0, index: 1)
    if withW {
        enc.setComputePipelineState(pW)
        enc.setBuffer(wRing, offset: (k % slabs) * slabBytes, index: 2)
        enc.setBuffer(bufN, offset: 0, index: 3)
        enc.setBuffer(bufPer, offset: 0, index: 4)
    } else {
        enc.setComputePipelineState(pTiny)
        enc.setBuffer(bufN, offset: 0, index: 2)
    }
}

struct Run { var cpu = 0.0; var gpu = 0.0; var wall = 0.0 }
func gpuSpan(_ cbs: [MTLCommandBuffer]) -> Double { cbs.last!.gpuEndTime - cbs.first!.gpuStartTime }

func runSerial(_ withW: Bool) -> Run {
    let t0 = CFAbsoluteTimeGetCurrent()
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder(dispatchType: .serial)!
    for k in 0..<K { bind(enc, k, withW); enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg) }
    enc.endEncoding(); cb.commit()
    let t1 = CFAbsoluteTimeGetCurrent()
    cb.waitUntilCompleted()
    return Run(cpu: t1 - t0, gpu: gpuSpan([cb]), wall: CFAbsoluteTimeGetCurrent() - t0)
}

func runMlx(_ withW: Bool, opsPerBuffer: Int = 400) -> Run {
    let t0 = CFAbsoluteTimeGetCurrent()
    var cbs: [MTLCommandBuffer] = []
    var k = 0
    while k < K {
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder(dispatchType: .concurrent)!
        for _ in 0..<min(opsPerBuffer, K - k) {
            bind(enc, k, withW)
            enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
            enc.memoryBarrier(scope: .buffers)
            k += 1
        }
        enc.endEncoding(); cb.commit(); cbs.append(cb)
    }
    let t1 = CFAbsoluteTimeGetCurrent()
    cbs.last!.waitUntilCompleted()
    return Run(cpu: t1 - t0, gpu: gpuSpan(cbs), wall: CFAbsoluteTimeGetCurrent() - t0)
}

func makeICB(_ withW: Bool) -> MTLIndirectCommandBuffer {
    let d = MTLIndirectCommandBufferDescriptor()
    d.commandTypes = .concurrentDispatch
    d.inheritBuffers = false
    d.inheritPipelineState = false
    d.maxKernelBufferBindCount = 5
    let icb = dev.makeIndirectCommandBuffer(descriptor: d, maxCommandCount: K, options: [])!
    for k in 0..<K {
        let c = icb.indirectComputeCommandAt(k)
        let (inp, out) = k % 2 == 0 ? (bufA, bufB) : (bufB, bufA)
        c.setKernelBuffer(inp, offset: 0, at: 0)
        c.setKernelBuffer(out, offset: 0, at: 1)
        if withW {
            c.setComputePipelineState(pW)
            c.setKernelBuffer(wRing, offset: (k % slabs) * slabBytes, at: 2)
            c.setKernelBuffer(bufN, offset: 0, at: 3)
            c.setKernelBuffer(bufPer, offset: 0, at: 4)
        } else {
            c.setComputePipelineState(pTiny)
            c.setKernelBuffer(bufN, offset: 0, at: 2)
        }
        c.concurrentDispatchThreadgroups(grid, threadsPerThreadgroup: tg)
        c.setBarrier()   // each command waits for the previous one
    }
    return icb
}

func runICB(_ icb: MTLIndirectCommandBuffer, _ withW: Bool) -> Run {
    let t0 = CFAbsoluteTimeGetCurrent()
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder(dispatchType: .concurrent)!
    enc.useResource(bufA, usage: [.read, .write])
    enc.useResource(bufB, usage: [.read, .write])
    enc.useResource(bufN, usage: .read)
    if withW { enc.useResource(wRing, usage: .read); enc.useResource(bufPer, usage: .read) }
    enc.executeCommandsInBuffer(icb, range: 0..<K)
    enc.endEncoding(); cb.commit()
    let t1 = CFAbsoluteTimeGetCurrent()
    cb.waitUntilCompleted()
    if let e = cb.error { print("ICB error:", e) }
    return Run(cpu: t1 - t0, gpu: gpuSpan([cb]), wall: CFAbsoluteTimeGetCurrent() - t0)
}

// The floor for "weights": stream the same bytes (K slabs) in one dispatch.
func runStream() -> Run {
    var count = UInt32(slabBytes * slabs / 16)
    let bc = dev.makeBuffer(bytes: &count, length: 4, options: .storageModeShared)!
    let threads = 32 * 1024
    let out = dev.makeBuffer(length: threads * 4, options: .storageModePrivate)!
    let passes = (K + slabs - 1) / slabs
    let t0 = CFAbsoluteTimeGetCurrent()
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pStream)
    enc.setBuffer(wRing, offset: 0, index: 0); enc.setBuffer(out, offset: 0, index: 1)
    enc.setBuffer(bc, offset: 0, index: 2)
    for _ in 0..<passes {
        enc.dispatchThreads(MTLSize(width: threads, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
    enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    let gpu = gpuSpan([cb]) * Double(K) / Double(passes * slabs)
    return Run(cpu: 0, gpu: gpu, wall: CFAbsoluteTimeGetCurrent() - t0)
}

func median(_ f: () -> Run) -> Run {
    _ = f(); _ = f()   // warm up
    var rs: [Run] = []
    for _ in 0..<reps { rs.append(f()) }
    func med(_ xs: [Double]) -> Double { xs.sorted()[xs.count / 2] }
    return Run(cpu: med(rs.map { $0.cpu }), gpu: med(rs.map { $0.gpu }), wall: med(rs.map { $0.wall }))
}

func report(_ name: String, _ r: Run) {
    print(String(format: "  %-8@ wall %7.2f ms   CPU encode %6.2f ms (%5.2f us/kernel)   GPU %7.2f ms (%5.2f us/kernel)",
                 name as NSString, r.wall * 1e3, r.cpu * 1e3, r.cpu * 1e6 / Double(K), r.gpu * 1e3, r.gpu * 1e6 / Double(K)))
}

print("\(dev.name): chain of \(K) dependent kernels, median of \(reps)")
for withW in [false, true] {
    print(withW ? "weights (2 MiB streamed per kernel, \(String(format: "%.2f", Double(K * slabBytes) / 1e9)) GB per chain):"
                : "tiny (10,240 floats per kernel):")
    report("serial", median { runSerial(withW) })
    report("mlx", median { runMlx(withW) })
    let t0 = CFAbsoluteTimeGetCurrent()
    let icb = makeICB(withW)
    print(String(format: "  (ICB encoded once in %.2f ms)", (CFAbsoluteTimeGetCurrent() - t0) * 1e3))
    report("icb", median { runICB(icb, withW) })
    if withW { report("one", median { runStream() }) }
}
