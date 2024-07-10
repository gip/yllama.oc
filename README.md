# yllama.oc

An on-chain implementation of inference for Llama 3 8b. 

The general intent (not totally achieved here) is to develop a very generic `yblock` canister easily deployable and configurable on the ICP. These `yblock` will be the building blocks on which AI algorithms can be uploaded and executed, achieving the goal of distributing the compute on a network of independant nodes. The consensus ensures the accuracy of the result even though we know nothing of the nodes. 

For this implementation, the load is distributed on 34 canisters. The code is currently not optimized. Next step will be to optimize by reducing overhead and make use of SIMD on wasm code.

The actual algorithm is implemented in [here](https://github.com/gip/yllama.rs).

## Building

To build you will need:
* This repositiry [yllama.oc](https://github.com/gip/yllama.oc). It depends on `yllama.rs`.
* The [yllama.rs](https://github.com/gip/yllama.rs). It depends on `tokenizers`.
* A patched version of the Hugging Face [tokenizers](https://github.com/huggingface/tokenizers) repo. The crates built on top of `getrandom` need to be modified as ICP code execution is deterministic and randomness is handled differently.

## Deploying and Running

To be updated

## Contact

gip.github@gmail.com
