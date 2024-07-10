use candid::CandidType;
use clap::Parser;
use half::f16;
use ic_agent::agent::http_transport::ReqwestTransport;
use ic_agent::export::Principal;
use ic_agent::identity::Secp256k1Identity;
use ic_agent::{Agent, Identity};
use ic_utils::call::AsyncCall;
use ic_utils::call::AsyncCaller;
use ic_utils::canister::{Canister, CanisterBuilder};
use ic_utils::interfaces::management_canister::ManagementCanister;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::time::Duration;
use tokio::time::sleep;

use yloader::*;
use ymath::tensor::*;

const EMBED: usize = 4096;
const VOCAB: usize = 128256;
const FF: usize = 14336;
const KV: usize = 1024;
const CONTEXT: usize = 2048;

const DELAY: core::time::Duration = Duration::from_secs(1);

#[derive(CandidType, Deserialize, Debug)]
struct Descr {
    name: String,
    shape: Vec<u32>,
}

async fn model_set_blocks(canister: &Canister<'_>, blocks: &Vec<Principal>) -> () {
    let caller: AsyncCaller<()> = canister
        .update("model_set_blocks")
        .with_args((blocks,))
        .build();
    let _: () = caller.call_and_wait().await.expect("call");
}

async fn upload_list(canister: &Canister<'_>) -> Vec<Descr> {
    let caller: AsyncCaller<(Vec<Descr>,)> = canister.update("upload_list").with_args(()).build();
    let (r,) = caller.call_and_wait().await.expect("call");
    r
}

async fn upload(canister: &Canister<'_>, name: String, new: bool, data: &[u8]) {
    let caller: AsyncCaller<()> = canister
        .update("upload")
        .with_args((name, new, data))
        .build();
    let () = caller.call_and_wait().await.expect("call");
}

async fn tensor_list(canister: &Canister<'_>) -> Vec<Descr> {
    let caller: AsyncCaller<(Vec<Descr>,)> = canister.update("tensor_list").with_args(()).build();
    let (r,) = caller.call_and_wait().await.expect("call");
    r
}

async fn tensor_new_vector(canister: &Canister<'_>, name: String, d0: usize) {
    println!("New vector {}", name);
    let caller: AsyncCaller<()> = canister
        .update("tensor_new_vector")
        .with_args((name, d0 as u32))
        .build();
    let _ = caller.call_and_wait().await.expect("call");
}

async fn tensor_new_matrix(canister: &Canister<'_>, name: String, d0: usize, d1: usize) {
    println!("New matrix {}", name);
    let caller: AsyncCaller<()> = canister
        .update("tensor_new_matrix")
        .with_args((name, d0 as u32, d1 as u32))
        .build();
    let _ = caller.call_and_wait().await.expect("call");
}

async fn tensor_load_matrix_from_row_fp16(
    canister: &Canister<'_>,
    name: &str,
    i: u32,
    vec: &[f16],
) {
    let mut attempts = 0;
    let max_attempts = 5;
    let vec: Vec<u16> = vec.iter().map(|v| v.to_bits()).collect();
    loop {
        let caller: AsyncCaller<()> = canister
            .update("tensor_load_matrix_from_row_fp16")
            .with_args((name, i, &vec))
            .build();
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
        let caller: AsyncCaller<()> = canister
            .update("tensor_load_matrix_from_row")
            .with_args((name, i, vec))
            .build();
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
        let caller: AsyncCaller<()> = canister
            .update("tensor_load_vector")
            .with_args((name, vec))
            .build();
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

async fn _model_block_forward(canister: &Canister<'_>, block: u32, x: &mut Vec<f32>, pos: u32) {
    let caller: AsyncCaller<(Vec<f32>,)> = canister
        .update("model_block_forward")
        .with_args((block, &*x, pos))
        .build();
    let (vec,): (Vec<f32>,) = caller.call_and_wait().await.expect("call");
    dbg!(vec.len());
    *x = vec
}

async fn model_forward(canister: &Canister<'_>, token: u32, pos: u32) -> u32 {
    let caller: AsyncCaller<(u32,)> = canister
        .update("model_forward")
        .with_args((token, pos))
        .build();
    let (predicted,): (u32,) = caller.call_and_wait().await.expect("call");
    predicted
}

async fn block_new_tensors(canister: &Canister<'_>, i: usize) {
    tensor_new_matrix(canister, format!("blk.{}.attn_q.weight", i), EMBED, EMBED).await;
    tensor_new_matrix(canister, format!("blk.{}.attn_k.weight", i), EMBED, KV).await;
    tensor_new_matrix(canister, format!("blk.{}.attn_v.weight", i), EMBED, KV).await;
    tensor_new_vector(canister, format!("blk.{}.attn_norm.weight", i), EMBED).await;
    tensor_new_matrix(
        canister,
        format!("blk.{}.attn_output.weight", i),
        EMBED,
        EMBED,
    )
    .await;

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

async fn load_data_matrix<const D0: usize, const D1: usize>(
    canister: &Canister<'_>,
    model: &ModelFile,
    name: &str,
    batch: usize,
) {
    assert!(D1 % batch == 0);
    let data = model
        .tensors
        .get(name)
        .expect(&format!("tensor '{}' not found", name));
    let dim = &data.dimensions;
    assert!(dim.len() == 2 && dim[0] as usize == D0 && dim[1] as usize == D1);
    let tensor32: Result<Tensor<false, f32, M<D0, D1>, MmapStore<f32, f32>>, _> =
        data.to_tensor(model);
    match tensor32 {
        Ok(tensor32) => {
            println!("Loading data for '{}' (fp32 mode)", name);
            let (_, slice) = tensor32.store;
            for i in (0..D1).step_by(batch) {
                print!("\rLoading for '{}' to {}/{}", name, i + batch, D1);
                let _ = std::io::stdout().flush();
                let row = &slice[i * D0..(i + batch) * D0];
                tensor_load_matrix_from_row(canister, &name, i as u32, row).await;
            }
            println!("\nLoading data for '{}'.. done", name);
        }
        Err(_) => {
            let batch = batch * 2;
            assert!(D1 % batch == 0);
            let tensor16: Tensor<false, f32, M<D0, D1>, MmapStore<f32, f16>> =
                data.to_tensor(model).expect("tensor is not f32 or f16");
            println!("Loading data for '{}' (fp16 mode)", name);
            let (_, slice) = tensor16.store;
            for i in (0..D1).step_by(batch) {
                print!("\rLoading for '{}' to {}/{}", name, i + batch, D1);
                let _ = std::io::stdout().flush();
                let row = &slice[i * D0..(i + batch) * D0];
                tensor_load_matrix_from_row_fp16(canister, &name, i as u32, row).await;
            }
            println!("\nLoading data for '{}'.. done", name);
        }
    }
}

async fn load_data_vector<const D0: usize>(canister: &Canister<'_>, model: &ModelFile, name: &str) {
    let data = model
        .tensors
        .get(name)
        .expect(&format!("tensor '{}' not found", name));
    let dim = &data.dimensions;
    assert!(dim.len() == 1 && dim[0] as usize == D0);
    let tensor: Tensor<false, f32, V<D0>, MmapStore<f32, f32>> =
        data.to_tensor(model).expect("tensor");
    println!("Loading data for '{}'", name);
    let (_, slice) = &tensor.store;
    tensor_load_vector(canister, &name, slice).await;
    println!("Loading data for '{}'.. done", name);
}

async fn block_load_tensors(canister: &Canister<'_>, model: &ModelFile, i: usize) {
    load_data_matrix::<EMBED, EMBED>(&canister, &model, &format!("blk.{}.attn_q.weight", i), 64)
        .await;
    load_data_matrix::<EMBED, KV>(&canister, &model, &format!("blk.{}.attn_k.weight", i), 64).await;
    load_data_matrix::<EMBED, KV>(&canister, &model, &format!("blk.{}.attn_v.weight", i), 64).await;
    load_data_vector::<EMBED>(&canister, &model, &format!("blk.{}.attn_norm.weight", i)).await;
    load_data_matrix::<EMBED, EMBED>(
        &canister,
        &model,
        &format!("blk.{}.attn_output.weight", i),
        64,
    )
    .await;

    load_data_matrix::<FF, EMBED>(&canister, &model, &format!("blk.{}.ffn_down.weight", i), 32)
        .await;
    load_data_matrix::<EMBED, FF>(&canister, &model, &format!("blk.{}.ffn_up.weight", i), 32).await;
    load_data_vector::<EMBED>(&canister, &model, &format!("blk.{}.ffn_norm.weight", i)).await;
    load_data_matrix::<EMBED, FF>(&canister, &model, &format!("blk.{}.ffn_gate.weight", i), 32)
        .await;
}

async fn model_input_new_tensors(canister: &Canister<'_>) {
    tensor_new_matrix(canister, "token_embd.weight".to_string(), EMBED, VOCAB).await;
}

async fn model_output_new_tensors(canister: &Canister<'_>) {
    tensor_new_matrix(canister, "output.weight".to_string(), EMBED, VOCAB).await;
    tensor_new_vector(canister, "output_norm.weight".to_string(), EMBED).await;
    tensor_new_vector(canister, "x2".to_string(), EMBED).await;
    tensor_new_vector(canister, "logits".to_string(), VOCAB).await;
}

async fn model_input_load_tensors(canister: &Canister<'_>, model: &ModelFile) {
    load_data_matrix::<EMBED, VOCAB>(&canister, &model, &format!("token_embd.weight"), 16).await;
}

async fn model_output_load_tensors(canister: &Canister<'_>, model: &ModelFile) {
    load_data_matrix::<EMBED, VOCAB>(&canister, &model, &format!("output.weight"), 16).await;
    load_data_vector::<EMBED>(&canister, &model, &format!("output_norm.weight")).await;
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(long)]
    ic: Option<Option<String>>,

    #[arg(short, long, default_value = "./")]
    dfx: String,

    #[arg(short, long)]
    file: String,

    #[arg(short, long)]
    block: usize,

    #[arg(long, default_value_t = 0)]
    pos: u32,

    #[arg(long)]
    token: Option<u32>,

    #[arg(long)]
    canister: String,

    #[arg(long, default_value_t = false)]
    list: bool,

    #[arg(long, default_value_t = false)]
    create: bool,

    #[arg(long, default_value_t = false)]
    load: bool,

    #[arg(long, default_value_t = false)]
    forward: bool,

    #[arg(long)]
    upload: Option<String>,

    #[arg(long)]
    set_blocks: Option<String>,

    #[arg(long)]
    set_controller: Option<String>,

    #[arg(long, default_value_t = false)]
    full: bool,
}

#[tokio::main]
async fn main() {
    println!("\n<<< ydriver\n");

    let args = Args::parse();

    // Canister Id
    let target = if args.ic.is_some() { "ic" } else { "local" };
    let canister_ids_file = if target == "local" {
        format!("{}/.dfx/local/canister_ids.json", args.dfx)
    } else {
        format!("{}/canister_ids.json", args.dfx)
    };
    println!("Canister file: {}", canister_ids_file);
    let content = fs::read_to_string(canister_ids_file).expect("canister_ids.json content");
    let json: serde_json::Value = serde_json::from_str(&content).expect("canister_ids.json file");
    let j0 = json.as_object().expect("Canister list in JSON");
    let canister_map: HashMap<&String, &str> = j0
        .iter()
        .map(|(k, c)| {
            (
                k,
                c.as_object()
                    .expect("object")
                    .get(target)
                    .expect("target")
                    .as_str()
                    .expect("canister name"),
            )
        })
        .collect();
    let canister_id: &str = match canister_map.get(&args.canister) {
        Some(s) => s,
        None => &args.canister,
    };

    let canister_principal = Principal::from_text(canister_id).expect("canister principal");
    println!("Canister Id: {}", canister_id);

    // Url
    let (_ic, url) = match args.ic {
        Some(_) => (true, "https://something.raw.icp0.io/"),
        None => (false, "http://127.0.0.1:4943/"),
    };

    // Model
    let (_, _, gguf) = load_fast(&args.file).expect("GGUF file");
    let model = load_build(&args.file, gguf).expect("GGUF model");

    // Agent & Canister
    let url = url;
    let transport = ReqwestTransport::create(url).expect("transport");
    let identity = Secp256k1Identity::from_pem_file("./identity.pem").expect("identity");
    let identity_principal = identity.sender().expect("identity principla");
    let mut agent = Agent::builder()
        .with_url(url)
        .with_identity(identity.clone())
        .build()
        .expect("agent");
    agent.set_transport(transport);
    let _ = agent.fetch_root_key().await;
    let canister = CanisterBuilder::new()
        .with_agent(&agent)
        .with_canister_id(canister_principal)
        .build()
        .expect("Canister");
    // println!("{:?}", identity.sender().expect("principal").to_text()); return;

    let manager = ManagementCanister::create(&agent);
    let (settings,) = manager
        .canister_status(&canister_principal)
        .call_and_wait()
        .await
        .expect("settings");
    for c in settings.settings.controllers {
        println!("Controller: {}", c);
    }

    let block = args.block;
    let pos: u32 = args.pos;

    if args.list {
        let l = tensor_list(&canister).await;
        println!("Tensors: {:?}", &l);
        let l = upload_list(&canister).await;
        println!("Uploads: {:?}", &l);
    }
    match args.upload {
        Some(str) => {
            let split: Vec<&str> = str.split("=").collect();
            assert!(split.len() == 2, "wrong upload format");
            let name = split[0];
            let file_path = split[1];
            let data: Vec<u8> = fs::read(file_path).expect("data file");
            let len = data.len();
            let mut c = 0;
            let b = 2_000_000;
            while c < len {
                upload(
                    &canister,
                    name.to_string(),
                    c == 0,
                    &data[c..std::cmp::min(c + b, len)],
                )
                .await;
                c += b;
            }
        }
        None => (),
    }
    if args.create {
        match block {
            100 => model_input_new_tensors(&canister).await,
            101 => model_output_new_tensors(&canister).await,
            _ => {
                assert!(block <= 31);
                block_new_tensors(&canister, block).await;
            }
        }
    }
    if args.load {
        match block {
            100 => model_input_load_tensors(&canister, &model).await,
            101 => model_output_load_tensors(&canister, &model).await,
            _ => {
                assert!(block <= 31);
                block_load_tensors(&canister, &model, block).await;
            }
        }
    }
    match args.set_blocks {
        None => (),
        Some(s) => {
            let split: Vec<_> = s.split(",").collect();
            let blocks: Vec<Principal> = split
                .iter()
                .map(|b| {
                    Principal::from_text(
                        canister_map
                            .get(&b.to_string())
                            .expect(&format!("canister {}", b)),
                    )
                    .expect("principal")
                })
                .collect();
            model_set_blocks(&canister, &blocks).await;
            for b in blocks {
                let builder = manager.update_settings(&b);
                let builder = builder.with_controller(canister_principal);
                assert!(identity_principal.to_text().starts_with("gi6m6"));
                let builder = builder.with_controller(identity_principal);
                builder.call_and_wait().await.expect("update_settings");
            }
        }
    }
    if args.forward {
        let token = args.token.expect("token");
        print!("Token input: {}", token);
        let _ = std::io::stdout().flush();
        let predicted = model_forward(&canister, token, pos).await;
        println!("\rToken input: {} predicted: {}", token, predicted);
    }

    println!("\n<<< done");
}
