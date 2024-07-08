use core::borrow;
use std::ops::Deref;
use std::{cell::RefCell, ops::DerefMut};
use candid::Principal;
use ic_cdk::caller;
use yllama::llama::*;
use yllama::llm::*;
use ymath::tensor::*;
use std::collections::HashMap;
use half::f16;

type YLlamaState<T> = HashMap<String, (Vec<u32>, RefCell<Vec<T>>)>;

thread_local! {
    static BLOCKS: RefCell<Vec<Principal>> = RefCell::default();
    static STORE: RefCell<YLlamaState<f32>> = RefCell::default();
}

#[ic_cdk::init]
fn init() {
    // Pass
}

#[ic_cdk::query]
fn tensor_get_vector(name: String) -> Vec<f32> {
    STORE.with(|store| {
        let mut borrow = store.borrow_mut();
        let borrow = borrow.deref_mut();
        let (shape, vec) = borrow.get_mut(&name).expect("tensor not found");
        if shape.len() != 1 {
            assert!(false, "Wrong shape")
        };
        let d0 = shape[0] as usize;
        let mut ret: Vec<f32> = Vec::with_capacity(d0);
        let vec = vec.borrow();
        for i in 0..d0 {
            ret.push(vec[i])
        };
        ret
    })
}

#[ic_cdk::query]
fn tensor_get_row_matrix(name: String, row: u32) -> Vec<f32> {
    let row = row as usize;
    STORE.with(|store| {
        let mut borrow = store.borrow_mut();
        let borrow = borrow.deref_mut();
        let (shape, vec) = borrow.get_mut(&name).expect("tensor not found");
        if shape.len() != 2 {
            assert!(false, "Wrong shape")
        };
        let d0 = shape[0] as usize;
        let d1 = shape[1] as usize;
        assert!(row < d1);
        let mut ret: Vec<f32> = Vec::with_capacity(d1);
        let vec = vec.borrow();
        for i in 0..d0 {
            ret.push(vec[row * d0 + i])
        };
        ret
    })
}

#[ic_cdk::update]
fn tensor_new_matrix(name: String, d0: u32, d1: u32) {
    STORE.with(|store| {
        let mut borrow = store.borrow_mut();
        let len = (d0 * d1) as usize;
        let vec = vec![0.0; len];
        borrow.insert(name, (vec![d0, d1], RefCell::new(vec)))
    });
}

#[ic_cdk::update]
fn tensor_new_vector(name: String, d0: u32) {
    STORE.with(|store| {
        let mut borrow = store.borrow_mut();
        let len = d0 as usize;
        let vec = vec![0.0; len];
        match borrow.get(&name) {
            Some(_) => (), // Already exists, do nothing
            None => { borrow.insert(name, (vec![d0], RefCell::new(vec))); }
        }
    });
}

#[ic_cdk::query]
fn status() -> String {
    let caller = ic_cdk::api::caller();
    ic_cdk::api::print(format!("Current caller in status is {}", caller.to_string()));
    let pages: usize = core::arch::wasm32::memory_size(0);
    format!("Wasm memory (64KB) pages: {}",  pages)
}

#[ic_cdk::update]
fn tensor_delete(name: String) {
    STORE.with(|store| {
        let mut borrow = store.borrow_mut();
        let borrow = borrow.deref_mut();
        if name == "ALL" {
            borrow.drain();
        } else {
            borrow.remove(&name);
        }
    });
}

#[ic_cdk::query]
fn tensor_list() -> Vec<(String, Vec<u32>)> {
    STORE.with(|store| {
        let borrow = store.borrow();
        borrow.iter().map(|(k, v)| (k.clone(), v.0.clone())).collect()
    })
}

#[ic_cdk::update]
fn tensor_load_matrix_from_row_fp16(name: String, row: u32, weights: Vec<u16>) -> () {
    let vec: Vec<f32> = weights.iter().map(|v| f16::from_bits(*v).to_f32() ).collect();
    tensor_load_matrix_from_row(name, row, vec);
}

#[ic_cdk::update]
fn tensor_load_matrix_from_row(name: String, row: u32, weights: Vec<f32>) -> () {
    let row = row as usize;
    STORE.with(|store| {
        let mut borrow = store.borrow_mut();
        let borrow = borrow.deref_mut();
        let (shape, vec) = match borrow.get_mut(&name) {
            None => panic!("Tensor '{}' not found", name),
            Some(x) => x

        };
        assert!(shape.len() == 2);
        let d0 = shape[0] as usize;
        let d1 = shape[1] as usize;
        assert!(row < d1);
        let mut vec = vec.borrow_mut();
        for i in 0..weights.len() {
            vec[row * d0 + i] = weights[i]
        };
        "".to_string() // Why is this required?
    });
}

#[ic_cdk::update]
fn tensor_load_vector(name: String, weights: Vec<f32>) -> () {
    STORE.with(|store| {
        let mut borrow = store.borrow_mut();
        let borrow = borrow.deref_mut();
        let (shape, vec) = match borrow.get_mut(&name) {
            None => panic!("Tensor '{}' not found", name),
            Some(x) => x

        };
        assert!(shape.len() == 1);
        let d0 = shape[0] as usize;
        assert!(weights.len() == d0);
        let mut vec = vec.borrow_mut();
        for i in 0..weights.len() {
            vec[i] = weights[i]
        };
        "".to_string() // Why is this required?
    });
}

//
struct ICP;

const EMBED: usize = 4096;
const VOCAB: usize = 128256;
const FF: usize = 14336;
const KV: usize = 1024;
const CONTEXT: usize = 2048;
type S<'a> = RefStore<'a, f32>;
type U<'a> = RefStore<'a, f32>;
type LlamaBlockType<'a> = LlamaBlock<'a, ICP, YLlamaState<f32>, f32, S<'a>, S<'a>, S<'a>, S<'a>, S<'a>, S<'a>, S<'a>, S<'a>, S<'a>, EMBED, VOCAB, FF, KV, CONTEXT, U<'a>, U<'a>, U<'a>, U<'a>, U<'a>, U<'a>, U<'a>, U<'a>>;

impl<'a, const RW: bool, const D0: usize> Instantiable<ICP, (&'a YLlamaState<f32>, String)> for Tensor<'a, RW, f32, V<D0>, RefStore<'a, f32>> {
    fn instantiate((state, name): (&'a YLlamaState<f32>, String)) -> Result<Self, anyhow::Error>
        where
            Self: Sized {
        let (_dim, vec) = state.get(&name).expect(&format!("tensor '{}' not found", name));
        Ok(Tensor {
            store: (0, vec)
        })
    }
}

impl<'a, const RW: bool, const D0: usize, const D1: usize> Instantiable<ICP, (&'a YLlamaState<f32>, String)> for Tensor<'a, RW, f32, M<D0, D1>, RefStore<'a, f32>> {
    fn instantiate((state, name): (&'a YLlamaState<f32>, String)) -> Result<Self, anyhow::Error>
        where
            Self: Sized {
        let (_dim, vec) = state.get(&name).expect(&format!("tensor '{}' not found", name));
        Ok(Tensor {
            store: (0, vec)
        })
    }
}

#[ic_cdk::update]
async fn model_set_blocks(new_blocks: Vec<Principal>) {
    BLOCKS.with(|blocks| {
        let mut borrow = blocks.borrow_mut();
        let borrow = borrow.deref_mut();
        *borrow = new_blocks;
    });
}

#[ic_cdk::update]
async fn model_get_blocks() -> Vec<Principal> {
    BLOCKS.with(|blocks| {
        let borrow = blocks.borrow();
        let borrow = borrow.deref();
        borrow.clone()
    })
}

#[ic_cdk::update]
async fn model_forward(x: Vec<f32>, pos: u32) -> Vec<f32> {
    let mut xx: Vec<f32> = x;
    let blocks = BLOCKS.with(|blocks| {
        let borrow = blocks.borrow();
        let borrow = borrow.deref();
        borrow.clone()
    });
    for (i, block) in blocks.iter().enumerate() {
        let (res,): (Vec<f32>,) = ic_cdk::call(*block, "model_block_forward", (i as u32, xx, pos)).await.expect("vector");
        xx = res
    };
    xx
}

#[ic_cdk::update]
async fn model_block_forward(i: u32, x: Vec<f32>, pos: u32) -> Vec<f32> {

    let params = LlamaParams {
        block_count: 32,
        _context_length: 8192,
        embedding_length: 4096,
        feed_forward_length: 14336,
        attention_head_count: 32,
        attention_head_count_kv: 8,
        attention_layer_norm_rms_epsilon: 1e-5,
        rope_freq_base: 500000.0,
        _rope_dimension_count: 128,
        vocab_size: 128256,
        _max_seq_len: 8192,
        _attention_kv_length: 1024,
    };

    let mut v = vec![0.0; 4096];
    for i in 0..x.len() {
        v[i] = x[i]
    }
    let mut xv: Tensor<true, f32, V<4096>, VecStore<f32>> = Tensor {
        store: v
    };
    STORE.with(|store| {
        let mut borrow = store.borrow_mut();
        let store = borrow.deref_mut();
        let store = &*store;
        let mut block: LlamaBlockType = Instantiable::instantiate((store, i as usize, params)).expect("instantiation failed");
        
        unsafe { block.forward(&mut xv, pos as usize) };
    });

    xv.store
}
