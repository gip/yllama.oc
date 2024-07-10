# yllama.oc

An on-chain implementation of inference for Llama 3 8b. 

## Overview

The project aims to create a generic yblock canister that's easily deployable and configurable on the Internet Computer Protocol (ICP). These yblock units serve as foundational components for uploading and executing AI algorithms, with the goal of distributing computation across a network of independent nodes. Consensus mechanisms ensure result accuracy without requiring knowledge of individual nodes.

In this implementation, the workload is distributed across 34 canisters. The code is currently unoptimized; future steps include reducing overhead and leveraging SIMD in WebAssembly.

The core algorithm is implemented in [here](https://github.com/gip/yllama.rs).

## Building

To build you will need:
* This repositiry [yllama.oc](https://github.com/gip/yllama.oc). It depends on `yllama.rs`.
* The [yllama.rs](https://github.com/gip/yllama.rs). It depends on `tokenizers`.
* A patched version of the Hugging Face [tokenizers](https://github.com/huggingface/tokenizers) repo. Crates built on getrandom need modification due to ICP's deterministic code execution and different randomness handling.

## Deploying and Running

[To be updated]

## Contact

gip.github@gmail.com
