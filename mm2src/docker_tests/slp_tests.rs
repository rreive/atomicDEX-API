use super::*;
use bitcoin_cash_slp::{slp_genesis_output, SlpTokenType};
use chain::TransactionOutput;
use coins::utxo::slp::{SlpToken, SlpUnspent};
use coins::utxo::utxo_common::{payment_script, send_outputs_from_my_address};
use coins::Transaction;
use keys::hash::H256;
use keys::{KeyPair, Private};
use script::{Builder, Opcode};

#[test]
fn mint_slp_token() {
    let (_ctx, coin, privkey) = generate_coin_with_random_privkey("MYCOIN", 1000.into());
    let private = Private {
        prefix: 0,
        secret: privkey.into(),
        compressed: true,
        checksum_type: Default::default(),
    };
    let keypair = KeyPair::from_private(private).unwrap();

    let output = slp_genesis_output(SlpTokenType::Fungible, "ADEX", "ADEX", "", "", 8, None, 1000_0000_0000);
    let script_pubkey = output.script.serialize().unwrap().to_vec().into();

    let op_return_output = TransactionOutput {
        value: output.value,
        script_pubkey,
    };
    let mint_output = TransactionOutput {
        value: 1000,
        script_pubkey: Builder::build_p2pkh(&keypair.public().address_hash()).to_bytes(),
    };
    let tx = send_outputs_from_my_address(coin.clone(), vec![op_return_output, mint_output])
        .wait()
        .unwrap();

    let slp = SlpToken::new(8, "ADEX".into(), tx.tx_hash().as_slice().into(), coin);
    let balance = slp.my_balance().wait().unwrap();
    let expected = BigDecimal::from(1000);
    assert_eq!(expected, balance.spendable);

    let secret = [0; 32];
    let secret_hash = dhash160(&secret);
    let time_lock = (now_ms() / 1000) as u32;
    let amount = 10000;

    let tx = block_on(slp.send_htlc(keypair.public(), time_lock, &*secret_hash, amount)).unwrap();
    println!("{}", hex::encode(tx.tx_hex()));

    let bch_unspent = UnspentInfo {
        outpoint: OutPoint {
            hash: H256::from(tx.tx_hash().0.as_slice()).reversed(),
            index: 1,
        },
        value: slp.dust(),
        height: None,
    };
    let slp_p2sh_utxo = SlpUnspent {
        bch_unspent,
        slp_amount: amount,
    };

    let redeem_script = payment_script(time_lock, &*secret_hash, keypair.public(), keypair.public());
    let script_data = Builder::default()
        .push_data(&secret)
        .push_opcode(Opcode::OP_0)
        .into_script();

    let spending_tx = block_on(slp.spend_p2sh(slp_p2sh_utxo, script_data, redeem_script)).unwrap();
    println!("spend hex {}", hex::encode(spending_tx.tx_hex()));
    println!("spend hash {}", hex::encode(spending_tx.tx_hash().0));
}
