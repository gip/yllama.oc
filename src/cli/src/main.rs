use std::io::Write;
use std::time::Duration;
use tokio::time::sleep;
use clap::Parser;
use ic_utils::canister::{Canister, CanisterBuilder};
use ic_utils::call::AsyncCaller;
use ic_agent::Agent;
use ic_agent::export::Principal;
use ic_agent::agent::http_transport::ReqwestTransport;
use ic_agent::identity::Secp256k1Identity;
use half::f16;

use ymath::tensor::*;
use yloader::*;

const EMBED: usize = 4096;
const VOCAB: usize = 128256;
const FF: usize = 14336;
const KV: usize = 1024;
const CONTEXT: usize = 2048;

const DELAY: core::time::Duration = Duration::from_secs(1);

async fn tensor_new_vector(canister: &Canister<'_>, name: String, d0: usize) {
    let caller: AsyncCaller<()> = 
        canister.update("tensor_new_vector").with_args((name, d0 as u32)).build();
    let _ = caller.call_and_wait().await.expect("call");
}

async fn tensor_new_matrix(canister: &Canister<'_>, name: String, d0: usize, d1: usize) {
    let caller: AsyncCaller<()> = 
        canister.update("tensor_new_matrix").with_args((name, d0 as u32, d1 as u32)).build();
    let _ = caller.call_and_wait().await.expect("call");
}

async fn tensor_load_matrix_from_row_fp16(canister: &Canister<'_>, name: &str, i: u32, vec: &[f16]) {
    let mut attempts = 0;
    let max_attempts = 5;
    let vec: Vec<u16> = vec.iter().map(|v| v.to_bits()).collect();
    loop {
        let caller: AsyncCaller<()> = 
            canister.update("tensor_load_matrix_from_row_fp16").with_args((name, i, &vec)).build();
        match caller.call_and_wait().await {
            Ok(_) => break,
            Err(_) => {
                attempts += 1;
                println!("error, retrying in {:?}", DELAY);
                if attempts >= max_attempts {
                    panic!("Failed after {} attempts", attempts);
                }
                sleep(DELAY * attempts).await;
            }
        }
    }
}

async fn tensor_load_matrix_from_row(canister: &Canister<'_>, name: &str, i: u32, vec: &[f32]) {
    let mut attempts = 0;
    let max_attempts = 5;
    loop {
        let caller: AsyncCaller<()> = 
            canister.update("tensor_load_matrix_from_row").with_args((name, i, vec)).build();
        match caller.call_and_wait().await {
            Ok(_) => break,
            Err(_) => {
                attempts += 1;
                println!("error, retrying in {:?}", DELAY);
                if attempts >= max_attempts {
                    panic!("Failed after {} attempts", attempts);
                }
                sleep(DELAY * attempts).await;
            }
        }
    }
}

async fn tensor_load_vector(canister: &Canister<'_>, name: &str, vec: &[f32]) {
    let mut attempts = 0;
    let max_attempts = 5;
    loop {
        let caller: AsyncCaller<()> = 
            canister.update("tensor_load_vector").with_args((name, vec)).build();
        match caller.call_and_wait().await {
            Ok(_) => break,
            Err(_) => {
                attempts += 1;
                println!("error, retrying in {:?}", DELAY);
                if attempts >= max_attempts {
                    panic!("Failed after {} attempts", attempts);
                }
                sleep(DELAY * attempts).await;
            }
        }
    }
}

async fn model_block_forward(canister: &Canister<'_>, block: u32, x: &mut Vec<f32>, pos: u32) {
    let caller: AsyncCaller<(Vec<f32>,)> = 
        canister.update("model_block_forward").with_args((block, &*x, pos)).build();
    let (vec,): (Vec<f32>,) = caller.call_and_wait().await.expect("call");
    dbg!(vec.len());
    dbg!(&vec);
    *x = vec
}

async fn model_forward(canister: &Canister<'_>, x: &mut Vec<f32>, pos: u32) {
    let caller: AsyncCaller<(Vec<f32>,)> = 
        canister.update("model_forward").with_args((&*x, pos)).build();
    let (vec,): (Vec<f32>,) = caller.call_and_wait().await.expect("call");
    dbg!(&vec);
    *x = vec
}

async fn block_new_tensors(canister: &Canister<'_>, i: usize) {
    tensor_new_matrix(canister, format!("blk.{}.attn_q.weight", i), EMBED, EMBED).await;
    tensor_new_matrix(canister, format!("blk.{}.attn_k.weight", i), EMBED, KV).await;
    tensor_new_matrix(canister, format!("blk.{}.attn_v.weight", i), EMBED, KV).await;
    tensor_new_vector(canister, format!("blk.{}.attn_norm.weight", i), EMBED).await;
    tensor_new_matrix(canister, format!("blk.{}.attn_output.weight", i), EMBED, EMBED).await;

    tensor_new_matrix(canister, format!("blk.{}.ffn_down.weight", i), FF, EMBED).await;
    tensor_new_matrix(canister, format!("blk.{}.ffn_up.weight", i), EMBED, FF).await;
    tensor_new_vector(canister, format!("blk.{}.ffn_norm.weight", i), EMBED).await;
    tensor_new_matrix(canister, format!("blk.{}.ffn_gate.weight", i), EMBED, FF).await;

    tensor_new_vector(canister, "xb".to_string(), EMBED).await;
    tensor_new_vector(canister, "xb2".to_string(), EMBED).await;
    tensor_new_vector(canister, "hb".to_string(), FF).await;
    tensor_new_vector(canister, "hb2".to_string(), FF).await;
    tensor_new_vector(canister, "q".to_string(), EMBED).await;
    tensor_new_matrix(canister, "k_cache".to_string(), FF, EMBED).await;
    tensor_new_matrix(canister, "v_cache".to_string(), FF, EMBED).await;
    tensor_new_matrix(canister, "attn_score".to_string(), CONTEXT, KV).await;
}

async fn load_data_matrix<const D0: usize, const D1: usize>(canister: &Canister<'_>, model: &ModelFile, name: &str, batch: usize) {
    assert!(D1 % batch == 0);
    let data = model.tensors.get(name).expect(&format!("tensor '{}' not found", name));
    let dim = &data.dimensions;
    assert!(dim.len() == 2 && dim[0] as usize == D0 && dim[1] as usize == D1);
    let tensor32: Result<Tensor<false, f32, M<D0, D1>, MmapStore<f32, f32>>, _> = data.to_tensor(model);
    match tensor32 {
        Ok(tensor32) => {
            println!("Loading data for '{}' (fp32 mode)", name);
            let (_, slice) = tensor32.store;
                for i in (0..D1).step_by(batch) {
                    print!("\rLoading for '{}' to {}/{}",name,  i+batch, D1);
                    let _ = std::io::stdout().flush();
                    let row = &slice[i * D0..(i + batch) * D0];
                    tensor_load_matrix_from_row(canister, &name, i as u32, row).await;
                }
            println!("\nLoading data for '{}'.. done", name);
        },
        Err(_) => {
            let batch = batch * 2;
            assert!(D1 % batch == 0);
            let tensor16: Tensor<false, f32, M<D0, D1>, MmapStore<f32, f16>> =
                data.to_tensor(model).expect("tensor is not f32 or f16");
                println!("Loading data for '{}' (fp16 mode)", name);
                let (_, slice) = tensor16.store;
                    for i in (0..D1).step_by(batch) {
                        print!("\rLoading for '{}' to {}/{}",name,  i+batch, D1);
                        let _ = std::io::stdout().flush();
                        let row = &slice[i * D0..(i + batch) * D0];
                        tensor_load_matrix_from_row_fp16(canister, &name, i as u32, row).await;
                    }
            println!("\nLoading data for '{}'.. done", name);
        }
    }
}

async fn load_data_vector<const D0: usize>(canister: &Canister<'_>, model: &ModelFile, name: &str) {
    let data = model.tensors.get(name).expect(&format!("tensor '{}' not found", name));
    let dim = &data.dimensions;
    assert!(dim.len() == 1 && dim[0] as usize == D0);
    let tensor: Tensor<false, f32, V<D0>, MmapStore<f32, f32>> = data.to_tensor(model).expect("tensor");
    println!("Loading data for '{}'", name);
    let (_, slice) = &tensor.store;
    tensor_load_vector(canister, &name, slice).await;
    println!("Loading data for '{}'.. done", name);
}

async fn block_load_tensors(canister: &Canister<'_>, model: &ModelFile, i: usize) {
    load_data_matrix::<EMBED, EMBED>(&canister, &model, &format!("blk.{}.attn_q.weight", i), 64).await;
    load_data_matrix::<EMBED, KV>(&canister, &model, &format!("blk.{}.attn_k.weight", i), 64).await;
    load_data_matrix::<EMBED, KV>(&canister, &model, &format!("blk.{}.attn_v.weight", i), 64).await;
    load_data_vector::<EMBED>(&canister, &model, &format!("blk.{}.attn_norm.weight", i)).await;
    load_data_matrix::<EMBED, EMBED>(&canister, &model, &format!("blk.{}.attn_output.weight", i), 64).await;

    load_data_matrix::<FF, EMBED>(&canister, &model, &format!("blk.{}.ffn_down.weight", i), 32).await;
    load_data_matrix::<EMBED, FF>(&canister, &model, &format!("blk.{}.ffn_up.weight", i), 32).await;
    load_data_vector::<EMBED>(&canister, &model, &format!("blk.{}.ffn_norm.weight", i)).await;
    load_data_matrix::<EMBED, FF>(&canister, &model, &format!("blk.{}.ffn_gate.weight", i), 32).await;
}

async fn model_new_tensors(canister: &Canister<'_>, model: &ModelFile) {
    tensor_new_matrix(canister, "token_embd.weight".to_string(), EMBED, VOCAB).await;
    tensor_new_matrix(canister, "output.weight".to_string(), EMBED, VOCAB).await;
    tensor_new_vector(canister, "output_norm.weight".to_string(), EMBED).await;
}

async fn model_load_tensors(canister: &Canister<'_>, model: &ModelFile) {
    load_data_matrix::<EMBED, VOCAB>(&canister, &model, &format!("token_embd.weight"), 16).await;
    load_data_matrix::<EMBED, VOCAB>(&canister, &model, &format!("output.weight"), 16).await;
    load_data_vector::<EMBED>(&canister, &model, &format!("output_norm.weight")).await;
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    file: String,

    #[arg(short, long)]
    block: usize,

    #[arg(long)]
    canister_id: String,

    #[arg(long)]
    canister_url: String,
}

#[tokio::main]
async fn main() {

    let args = Args::parse();

    let (_, _, gguf) = load_fast(&args.file).expect("GGUF file");
    let model = load_build(&args.file, gguf).expect("GGUF model");

    let canister_id = args.canister_id; // "bkyz2-fmaaa-aaaaa-qaaaq-cai"; // Local
    //let canister_id = "htqzs-pqaaa-aaaap-ahndq-cai"; // IC
    let canister_principal = Principal::from_text(canister_id).expect("canister principal");

    // Agent & Canister
    let url = args.canister_url; // "http://127.0.0.1:4943/?canisterId=bkyz2-fmaaa-aaaaa-qaaaq-cai";
    //let url = "https://htqzs-pqaaa-aaaap-ahndq-cai.raw.icp0.io/";
    let transport = ReqwestTransport::create(&url).expect("transport");
    let identity = Secp256k1Identity::from_pem_file("./identity.pem").expect("identity");
    let mut agent = Agent::builder()
        .with_url(&url)
        .with_identity(identity)
        .build().expect("agent");
    agent.set_transport(transport);
    let _ = agent.fetch_root_key().await;
    let canister = CanisterBuilder::new()
        .with_agent(&agent)
        .with_canister_id(canister_principal)
        .build()
        .expect("Canister");

    let block = args.block;
    let pos: u32 = 0;

    // block_new_tensors(&canister, block).await;
    block_load_tensors(&canister,&model, block).await;

    let mut x: Vec<f32> = vec![0.4; EMBED as usize];
    // model_forward(&canister, &mut x, pos).await;

    //model_block_forward(&canister, 0,  &mut x, pos).await;
    //dbg!(x);

    println!("done")
}