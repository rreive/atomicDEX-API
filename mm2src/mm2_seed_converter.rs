use sha2::{Sha256, Digest};
use hex;

#[inline]
pub fn sha256(input: &[u8]) -> Vec<u8> {
    let mut hasher = Sha256::new();
    hasher.input(input);
    hasher.result().to_vec()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        println!("Program accepts exactly 1 arg!");
        println!("Usage: seed-converter \"your seed phrase\"");
        std::process::exit(1);
    }
    let seed = &args[1];

    let mut hash = sha256(seed.as_bytes());
    hash[0] &= 248;
    hash[31] &= 127;
    hash[31] |= 64;

    println!("Your ETH privkey derived from seed:");
    println!("0x{}", hex::encode(hash));
}
