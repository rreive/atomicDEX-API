#![allow(dead_code)]
#![allow(unused_variables)]

use super::utxo_standard::UtxoStandardCoin;
use crate::utxo::rpc_clients::{UnspentInfo, UtxoRpcClientEnum, UtxoRpcError};
use crate::utxo::utxo_common::{big_decimal_from_sat_unsigned, payment_script};
use crate::utxo::{generate_and_send_tx, FeePolicy, RecentlySpentOutPoints, UtxoCommonOps, UtxoTx};
use crate::{BalanceError, BalanceFut, CoinBalance, FeeApproxStage, FoundSwapTxSpend, HistorySyncState, MarketCoinOps,
            MmCoin, NegotiateSwapContractAddrErr, SwapOps, TradeFee, TradePreimageFut, TradePreimageValue,
            TransactionEnum, TransactionFut, ValidateAddressResult, WithdrawFut, WithdrawRequest};
use bitcoin_cash_slp::{slp_send_output, SlpTokenType, TokenId};
use bitcrypto::dhash160;
use chain::TransactionOutput;
use common::mm_ctx::MmArc;
use common::mm_error::prelude::*;
use common::mm_number::{BigDecimal, MmNumber};
use derive_more::Display;
use futures::compat::Future01CompatExt;
use futures::lock::MutexGuard as AsyncMutexGuard;
use futures::{FutureExt, TryFutureExt};
use futures01::Future;
use keys::Public;
use primitives::hash::H256;
use rpc::v1::types::Bytes as BytesJson;
use script::bytes::Bytes;
use script::{Builder as ScriptBuilder, Opcode};
use serde_json::Value as Json;
use serialization::{deserialize, Deserializable, Error, Reader};
use serialization_derive::Deserializable;
use std::convert::TryInto;

#[derive(Clone, Debug)]
pub struct SlpToken {
    decimals: u8,
    ticker: String,
    token_id: H256,
    platform_utxo: UtxoStandardCoin,
}

struct SlpUnspent {
    bch_unspent: UnspentInfo,
    slp_amount: u64,
}

struct SlpOutput {
    amount: u64,
    script_pubkey: Bytes,
}

/// The SLP transaction preimage
struct SlpTxPreimage<'a> {
    inputs: Vec<UnspentInfo>,
    outputs: Vec<TransactionOutput>,
    recently_spent: AsyncMutexGuard<'a, RecentlySpentOutPoints>,
}

impl SlpToken {
    pub fn new(decimals: u8, ticker: String, token_id: H256, platform_utxo: UtxoStandardCoin) -> SlpToken {
        SlpToken {
            decimals,
            ticker,
            token_id,
            platform_utxo,
        }
    }

    fn rpc(&self) -> &UtxoRpcClientEnum { &self.platform_utxo.as_ref().rpc_client }

    /// Returns unspents of the SLP token plus plain BCH UTXOs plus RecentlySpentOutPoints mutex guard
    async fn slp_unspents(
        &self,
    ) -> Result<
        (
            Vec<SlpUnspent>,
            Vec<UnspentInfo>,
            AsyncMutexGuard<'_, RecentlySpentOutPoints>,
        ),
        MmError<SlpUnspentsErr>,
    > {
        let (unspents, recently_spent) = self
            .platform_utxo
            .list_unspent_ordered(&self.platform_utxo.as_ref().my_address)
            .await?;

        let mut slp_unspents = vec![];
        let mut bch_unspents = vec![];

        for unspent in unspents {
            let prev_tx_bytes = self
                .rpc()
                .get_transaction_bytes(unspent.outpoint.hash.reversed().into())
                .compat()
                .await?;
            let prev_tx: UtxoTx = deserialize(prev_tx_bytes.0.as_slice()).map_to_mm(SlpUnspentsErr::from)?;
            match parse_slp_script(&prev_tx.outputs[0].script_pubkey) {
                Ok(slp_data) => match slp_data.transaction {
                    SlpTransaction::Send { token_id, amounts } => {
                        if H256::from(token_id.as_slice()) == self.token_id {
                            match amounts.get(unspent.outpoint.index as usize - 1) {
                                Some(slp_amount) => slp_unspents.push(SlpUnspent {
                                    bch_unspent: unspent,
                                    slp_amount: *slp_amount,
                                }),
                                None => bch_unspents.push(unspent),
                            }
                        }
                    },
                    SlpTransaction::Genesis {
                        initial_token_mint_quantity,
                        ..
                    } => {
                        if prev_tx.hash().reversed() == self.token_id
                            && initial_token_mint_quantity.len() == 8
                            && unspent.outpoint.index == 1
                        {
                            let slp_amount = u64::from_be_bytes(initial_token_mint_quantity.try_into().unwrap());
                            slp_unspents.push(SlpUnspent {
                                bch_unspent: unspent,
                                slp_amount,
                            });
                        } else {
                            bch_unspents.push(unspent)
                        }
                    },
                    SlpTransaction::Mint {
                        token_id,
                        additional_token_quantity,
                        ..
                    } => {
                        if H256::from(token_id.as_slice()) == self.token_id && additional_token_quantity.len() == 8 {
                            let slp_amount = u64::from_be_bytes(additional_token_quantity.try_into().unwrap());
                            slp_unspents.push(SlpUnspent {
                                bch_unspent: unspent,
                                slp_amount,
                            });
                        }
                    },
                },
                Err(_) => bch_unspents.push(unspent),
            }
        }

        slp_unspents.sort_by(|a, b| a.slp_amount.cmp(&b.slp_amount));
        Ok((slp_unspents, bch_unspents, recently_spent))
    }

    /// Generates the inputs and outputs set to spend the SLP from my address to the desired destinations (script pubkeys)
    async fn generate_slp_tx_preimage(
        &self,
        slp_outputs: Vec<SlpOutput>,
    ) -> Result<SlpTxPreimage<'_>, MmError<GenSlpSpendErr>> {
        let (slp_unspents, bch_unspents, recently_spent) = self.slp_unspents().await?;
        let total_slp_output = slp_outputs.iter().fold(0, |cur, slp_out| cur + slp_out.amount);
        let mut total_slp_input = 0;

        let mut inputs = vec![];
        for slp_utxo in slp_unspents {
            if total_slp_input >= total_slp_output {
                break;
            }

            total_slp_input += slp_utxo.slp_amount;
            inputs.push(slp_utxo.bch_unspent);
        }

        if total_slp_input < total_slp_output {
            return MmError::err(GenSlpSpendErr::InsufficientSlpBalance);
        }
        let change = total_slp_input - total_slp_output;

        inputs.extend(bch_unspents);

        let mut amounts_for_op_return: Vec<_> = slp_outputs.iter().map(|spend_to| spend_to.amount).collect();
        if change > 0 {
            amounts_for_op_return.push(change);
        }

        // TODO generate the script in MM2 instead of using the external library
        let op_return_out = slp_send_output(
            SlpTokenType::Fungible,
            &TokenId::from_slice(&*self.token_id).unwrap(),
            &amounts_for_op_return,
        );
        let op_return_out_mm = TransactionOutput {
            value: 0,
            script_pubkey: op_return_out.script.serialize().unwrap().to_vec().into(),
        };
        let mut outputs = vec![op_return_out_mm];

        outputs.extend(slp_outputs.into_iter().map(|spend_to| TransactionOutput {
            value: self.platform_utxo.as_ref().dust_amount,
            script_pubkey: spend_to.script_pubkey,
        }));

        if change > 0 {
            let slp_change_out = TransactionOutput {
                value: self.platform_utxo.as_ref().dust_amount,
                script_pubkey: ScriptBuilder::build_p2pkh(&self.platform_utxo.my_public_key().address_hash())
                    .to_bytes(),
            };
            outputs.push(slp_change_out);
        }

        Ok(SlpTxPreimage {
            inputs,
            outputs,
            recently_spent,
        })
    }

    pub async fn send_htlc(
        &self,
        other_pub: &Public,
        time_lock: u32,
        secret_hash: &[u8],
        amount: u64,
    ) -> Result<UtxoTx, String> {
        let payment_script = payment_script(time_lock, secret_hash, self.platform_utxo.my_public_key(), other_pub);
        let script_pubkey = ScriptBuilder::build_p2sh(&dhash160(&payment_script)).to_bytes();
        let slp_out = SlpOutput { amount, script_pubkey };
        let preimage = try_s!(self.generate_slp_tx_preimage(vec![slp_out]).await);
        generate_and_send_tx(
            &self.platform_utxo,
            preimage.inputs,
            preimage.outputs,
            FeePolicy::SendExact,
            preimage.recently_spent,
        )
        .await
    }
}

/// https://slp.dev/specs/slp-token-type-1/#transaction-detail
#[derive(Debug, Eq, PartialEq)]
enum SlpTransaction {
    /// https://slp.dev/specs/slp-token-type-1/#genesis-token-genesis-transaction
    Genesis {
        token_ticker: String,
        token_name: String,
        token_document_url: String,
        token_document_hash: Vec<u8>,
        decimals: Vec<u8>,
        mint_baton_vout: Vec<u8>,
        initial_token_mint_quantity: Vec<u8>,
    },
    /// https://slp.dev/specs/slp-token-type-1/#mint-extended-minting-transaction
    Mint {
        token_id: Vec<u8>,
        mint_baton_vout: Vec<u8>,
        additional_token_quantity: Vec<u8>,
    },
    /// https://slp.dev/specs/slp-token-type-1/#send-spend-transaction
    Send { token_id: Vec<u8>, amounts: Vec<u64> },
}

impl Deserializable for SlpTransaction {
    fn deserialize<T>(reader: &mut Reader<T>) -> Result<Self, Error>
    where
        Self: Sized,
        T: std::io::Read,
    {
        let transaction_type: String = reader.read()?;
        match transaction_type.as_str() {
            "GENESIS" => {
                let token_ticker = reader.read()?;
                let token_name = reader.read()?;
                let maybe_push_op_code: u8 = reader.read()?;
                let token_document_url = if maybe_push_op_code == Opcode::OP_PUSHDATA1 as u8 {
                    reader.read()?
                } else {
                    let mut url = vec![0; maybe_push_op_code as usize];
                    reader.read_slice(&mut url)?;
                    String::from_utf8(url).map_err(|e| Error::Custom(e.to_string()))?
                };

                let maybe_push_op_code: u8 = reader.read()?;
                let token_document_hash = if maybe_push_op_code == Opcode::OP_PUSHDATA1 as u8 {
                    reader.read_list()?
                } else {
                    let mut hash = vec![0; maybe_push_op_code as usize];
                    reader.read_slice(&mut hash)?;
                    hash
                };
                let decimals = reader.read_list()?;
                let maybe_push_op_code: u8 = reader.read()?;
                let mint_baton_vout = if maybe_push_op_code == Opcode::OP_PUSHDATA1 as u8 {
                    reader.read_list()?
                } else {
                    let mut baton = vec![0; maybe_push_op_code as usize];
                    reader.read_slice(&mut baton)?;
                    baton
                };
                let initial_token_mint_quantity = reader.read_list()?;

                Ok(SlpTransaction::Genesis {
                    token_ticker,
                    token_name,
                    token_document_url,
                    token_document_hash,
                    decimals,
                    mint_baton_vout,
                    initial_token_mint_quantity,
                })
            },
            "MINT" => Ok(SlpTransaction::Mint {
                token_id: reader.read_list()?,
                mint_baton_vout: reader.read_list()?,
                additional_token_quantity: reader.read_list()?,
            }),
            "SEND" => {
                let token_id = reader.read_list()?;
                let mut amounts = Vec::with_capacity(1);
                while !reader.is_finished() {
                    let bytes: Vec<u8> = reader.read_list()?;
                    if bytes.len() != 8 {
                        return Err(Error::Custom(format!("Expected 8 bytes, got {}", bytes.len())));
                    }
                    let amount = u64::from_be_bytes(bytes.try_into().expect("length is 8 bytes"));
                    amounts.push(amount)
                }

                Ok(SlpTransaction::Send { token_id, amounts })
            },
            _ => Err(Error::Custom(format!(
                "Unsupported transaction type {}",
                transaction_type
            ))),
        }
    }
}

#[derive(Deserializable)]
struct SlpTxDetails {
    op_code: u8,
    lokad_id: String,
    token_type: String,
    transaction: SlpTransaction,
}

#[derive(Debug)]
enum ParseSlpScriptError {
    NotOpReturn,
    NotSlp,
    DeserializeFailed(Error),
}

impl From<Error> for ParseSlpScriptError {
    fn from(err: Error) -> ParseSlpScriptError { ParseSlpScriptError::DeserializeFailed(err) }
}

fn parse_slp_script(script: &[u8]) -> Result<SlpTxDetails, MmError<ParseSlpScriptError>> {
    let details: SlpTxDetails = deserialize(script).map_to_mm(ParseSlpScriptError::from)?;
    if Opcode::from_u8(details.op_code) != Some(Opcode::OP_RETURN) {
        return MmError::err(ParseSlpScriptError::NotOpReturn);
    }
    Ok(details)
}

#[derive(Debug, Display)]
enum SlpUnspentsErr {
    RpcError(UtxoRpcError),
    #[display(fmt = "TxDeserializeError: {:?}", _0)]
    TxDeserializeError(Error),
}

impl From<UtxoRpcError> for SlpUnspentsErr {
    fn from(err: UtxoRpcError) -> SlpUnspentsErr { SlpUnspentsErr::RpcError(err) }
}

impl From<Error> for SlpUnspentsErr {
    fn from(err: Error) -> SlpUnspentsErr { SlpUnspentsErr::TxDeserializeError(err) }
}

impl From<SlpUnspentsErr> for BalanceError {
    fn from(err: SlpUnspentsErr) -> BalanceError {
        match err {
            SlpUnspentsErr::RpcError(e) => BalanceError::Transport(e.to_string()),
            SlpUnspentsErr::TxDeserializeError(e) => BalanceError::Internal(format!("{:?}", e)),
        }
    }
}

#[derive(Debug, Display)]
enum GenSlpSpendErr {
    GetUnspentsErr(SlpUnspentsErr),
    InsufficientSlpBalance,
}

impl From<SlpUnspentsErr> for GenSlpSpendErr {
    fn from(err: SlpUnspentsErr) -> GenSlpSpendErr { GenSlpSpendErr::GetUnspentsErr(err) }
}

impl MarketCoinOps for SlpToken {
    fn ticker(&self) -> &str { &self.ticker }

    fn my_address(&self) -> Result<String, String> { unimplemented!() }

    fn my_balance(&self) -> BalanceFut<CoinBalance> {
        let coin = self.clone();
        let fut = async move {
            let (slp_unspents, _, _) = coin.slp_unspents().await?;
            let spendable_sat = slp_unspents.iter().fold(0, |cur, unspent| cur + unspent.slp_amount);
            let spendable = big_decimal_from_sat_unsigned(spendable_sat, coin.decimals);
            Ok(CoinBalance {
                spendable,
                unspendable: 0.into(),
            })
        };
        Box::new(fut.boxed().compat())
    }

    fn base_coin_balance(&self) -> BalanceFut<BigDecimal> { unimplemented!() }

    /// Receives raw transaction bytes in hexadecimal format as input and returns tx hash in hexadecimal format
    fn send_raw_tx(&self, tx: &str) -> Box<dyn Future<Item = String, Error = String> + Send> { unimplemented!() }

    fn wait_for_confirmations(
        &self,
        tx: &[u8],
        confirmations: u64,
        requires_nota: bool,
        wait_until: u64,
        check_every: u64,
    ) -> Box<dyn Future<Item = (), Error = String> + Send> {
        unimplemented!()
    }

    fn wait_for_tx_spend(
        &self,
        transaction: &[u8],
        wait_until: u64,
        from_block: u64,
        swap_contract_address: &Option<BytesJson>,
    ) -> TransactionFut {
        unimplemented!()
    }

    fn tx_enum_from_bytes(&self, bytes: &[u8]) -> Result<TransactionEnum, String> { unimplemented!() }

    fn current_block(&self) -> Box<dyn Future<Item = u64, Error = String> + Send> { unimplemented!() }

    fn address_from_pubkey_str(&self, pubkey: &str) -> Result<String, String> { unimplemented!() }

    fn display_priv_key(&self) -> String { unimplemented!() }

    fn min_tx_amount(&self) -> BigDecimal { unimplemented!() }

    fn min_trading_vol(&self) -> MmNumber { MmNumber::from("0.00777") }
}

impl SwapOps for SlpToken {
    fn send_taker_fee(&self, fee_addr: &[u8], amount: BigDecimal) -> TransactionFut { unimplemented!() }

    fn send_maker_payment(
        &self,
        time_lock: u32,
        taker_pub: &[u8],
        secret_hash: &[u8],
        amount: BigDecimal,
        swap_contract_address: &Option<BytesJson>,
    ) -> TransactionFut {
        unimplemented!()
    }

    fn send_taker_payment(
        &self,
        time_lock: u32,
        maker_pub: &[u8],
        secret_hash: &[u8],
        amount: BigDecimal,
        swap_contract_address: &Option<BytesJson>,
    ) -> TransactionFut {
        unimplemented!()
    }

    fn send_maker_spends_taker_payment(
        &self,
        taker_payment_tx: &[u8],
        time_lock: u32,
        taker_pub: &[u8],
        secret: &[u8],
        swap_contract_address: &Option<BytesJson>,
    ) -> TransactionFut {
        unimplemented!()
    }

    fn send_taker_spends_maker_payment(
        &self,
        maker_payment_tx: &[u8],
        time_lock: u32,
        maker_pub: &[u8],
        secret: &[u8],
        swap_contract_address: &Option<BytesJson>,
    ) -> TransactionFut {
        unimplemented!()
    }

    fn send_taker_refunds_payment(
        &self,
        taker_payment_tx: &[u8],
        time_lock: u32,
        maker_pub: &[u8],
        secret_hash: &[u8],
        swap_contract_address: &Option<BytesJson>,
    ) -> TransactionFut {
        unimplemented!()
    }

    fn send_maker_refunds_payment(
        &self,
        maker_payment_tx: &[u8],
        time_lock: u32,
        taker_pub: &[u8],
        secret_hash: &[u8],
        swap_contract_address: &Option<BytesJson>,
    ) -> TransactionFut {
        unimplemented!()
    }

    fn validate_fee(
        &self,
        fee_tx: &TransactionEnum,
        expected_sender: &[u8],
        fee_addr: &[u8],
        amount: &BigDecimal,
        min_block_number: u64,
    ) -> Box<dyn Future<Item = (), Error = String> + Send> {
        unimplemented!()
    }

    fn validate_maker_payment(
        &self,
        payment_tx: &[u8],
        time_lock: u32,
        maker_pub: &[u8],
        priv_bn_hash: &[u8],
        amount: BigDecimal,
        swap_contract_address: &Option<BytesJson>,
    ) -> Box<dyn Future<Item = (), Error = String> + Send> {
        unimplemented!()
    }

    fn validate_taker_payment(
        &self,
        payment_tx: &[u8],
        time_lock: u32,
        taker_pub: &[u8],
        priv_bn_hash: &[u8],
        amount: BigDecimal,
        swap_contract_address: &Option<BytesJson>,
    ) -> Box<dyn Future<Item = (), Error = String> + Send> {
        unimplemented!()
    }

    fn check_if_my_payment_sent(
        &self,
        time_lock: u32,
        other_pub: &[u8],
        secret_hash: &[u8],
        search_from_block: u64,
        swap_contract_address: &Option<BytesJson>,
    ) -> Box<dyn Future<Item = Option<TransactionEnum>, Error = String> + Send> {
        unimplemented!()
    }

    fn search_for_swap_tx_spend_my(
        &self,
        time_lock: u32,
        other_pub: &[u8],
        secret_hash: &[u8],
        tx: &[u8],
        search_from_block: u64,
        swap_contract_address: &Option<BytesJson>,
    ) -> Result<Option<FoundSwapTxSpend>, String> {
        unimplemented!()
    }

    fn search_for_swap_tx_spend_other(
        &self,
        time_lock: u32,
        other_pub: &[u8],
        secret_hash: &[u8],
        tx: &[u8],
        search_from_block: u64,
        swap_contract_address: &Option<BytesJson>,
    ) -> Result<Option<FoundSwapTxSpend>, String> {
        unimplemented!()
    }

    fn extract_secret(&self, secret_hash: &[u8], spend_tx: &[u8]) -> Result<Vec<u8>, String> { unimplemented!() }

    fn negotiate_swap_contract_addr(
        &self,
        other_side_address: Option<&[u8]>,
    ) -> Result<Option<BytesJson>, MmError<NegotiateSwapContractAddrErr>> {
        Ok(None)
    }
}

impl MmCoin for SlpToken {
    fn is_asset_chain(&self) -> bool { unimplemented!() }

    fn withdraw(&self, req: WithdrawRequest) -> WithdrawFut { unimplemented!() }

    fn decimals(&self) -> u8 { unimplemented!() }

    fn convert_to_address(&self, from: &str, to_address_format: Json) -> Result<String, String> { unimplemented!() }

    fn validate_address(&self, address: &str) -> ValidateAddressResult { unimplemented!() }

    fn process_history_loop(&self, ctx: MmArc) -> Box<dyn Future<Item = (), Error = ()> + Send> { unimplemented!() }

    fn history_sync_status(&self) -> HistorySyncState { unimplemented!() }

    /// Get fee to be paid per 1 swap transaction
    fn get_trade_fee(&self) -> Box<dyn Future<Item = TradeFee, Error = String> + Send> { unimplemented!() }

    fn get_sender_trade_fee(&self, value: TradePreimageValue, stage: FeeApproxStage) -> TradePreimageFut<TradeFee> {
        unimplemented!()
    }

    fn get_receiver_trade_fee(&self, stage: FeeApproxStage) -> TradePreimageFut<TradeFee> { unimplemented!() }

    fn get_fee_to_send_taker_fee(
        &self,
        dex_fee_amount: BigDecimal,
        stage: FeeApproxStage,
    ) -> TradePreimageFut<TradeFee> {
        unimplemented!()
    }

    fn required_confirmations(&self) -> u64 { 1 }

    fn requires_notarization(&self) -> bool { false }

    fn set_required_confirmations(&self, _confirmations: u64) { unimplemented!() }

    fn set_requires_notarization(&self, _requires_nota: bool) { unimplemented!() }

    fn swap_contract_address(&self) -> Option<BytesJson> { unimplemented!() }

    fn mature_confirmations(&self) -> Option<u32> { unimplemented!() }
}

// https://slp.dev/specs/slp-token-type-1/#examples
#[test]
fn test_parse_slp_script() {
    // Send single output
    let script = hex::decode("6a04534c500001010453454e4420e73b2b28c14db8ebbf97749988b539508990e1708021067f206f49d55807dbf4080000000005f5e100").unwrap();
    let slp_data = parse_slp_script(&script).unwrap();
    assert_eq!(slp_data.lokad_id, "SLP\0");
    let expected_amount = 100000000u64;
    let expected_transaction = SlpTransaction::Send {
        token_id: hex::decode("e73b2b28c14db8ebbf97749988b539508990e1708021067f206f49d55807dbf4").unwrap(),
        amounts: vec![expected_amount],
    };

    assert_eq!(expected_transaction, slp_data.transaction);

    // Genesis
    let script =
        hex::decode("6a04534c500001010747454e45534953044144455804414445584c004c0001084c0008000000174876e800").unwrap();
    let slp_data = parse_slp_script(&script).unwrap();
    assert_eq!(slp_data.lokad_id, "SLP\0");
    let initial_token_mint_quantity = 1000_0000_0000u64.to_be_bytes().to_vec();
    let expected_transaction = SlpTransaction::Genesis {
        token_ticker: "ADEX".to_string(),
        token_name: "ADEX".to_string(),
        token_document_url: "".to_string(),
        token_document_hash: vec![],
        decimals: vec![8],
        mint_baton_vout: vec![],
        initial_token_mint_quantity,
    };

    assert_eq!(expected_transaction, slp_data.transaction);

    // Genesis from docs example
    let script =
        hex::decode("6a04534c500001010747454e45534953045553445423546574686572204c74642e20555320646f6c6c6172206261636b656420746f6b656e734168747470733a2f2f7465746865722e746f2f77702d636f6e74656e742f75706c6f6164732f323031362f30362f546574686572576869746550617065722e70646620db4451f11eda33950670aaf59e704da90117ff7057283b032cfaec77793139160108010208002386f26fc10000").unwrap();
    let slp_data = parse_slp_script(&script).unwrap();
    assert_eq!(slp_data.lokad_id, "SLP\0");
    let initial_token_mint_quantity = 10000000000000000u64.to_be_bytes().to_vec();
    let expected_transaction = SlpTransaction::Genesis {
        token_ticker: "USDT".to_string(),
        token_name: "Tether Ltd. US dollar backed tokens".to_string(),
        token_document_url: "https://tether.to/wp-content/uploads/2016/06/TetherWhitePaper.pdf".to_string(),
        token_document_hash: hex::decode("db4451f11eda33950670aaf59e704da90117ff7057283b032cfaec7779313916").unwrap(),
        decimals: vec![8],
        mint_baton_vout: vec![2],
        initial_token_mint_quantity,
    };

    assert_eq!(expected_transaction, slp_data.transaction);

    // Mint
    let script =
        hex::decode("6a04534c50000101044d494e5420550d19eb820e616a54b8a73372c4420b5a0567d8dc00f613b71c5234dc884b35010208002386f26fc10000").unwrap();
    let slp_data = parse_slp_script(&script).unwrap();
    assert_eq!(slp_data.lokad_id, "SLP\0");
    let expected_transaction = SlpTransaction::Mint {
        token_id: hex::decode("550d19eb820e616a54b8a73372c4420b5a0567d8dc00f613b71c5234dc884b35").unwrap(),
        mint_baton_vout: vec![2],
        additional_token_quantity: hex::decode("002386f26fc10000").unwrap(),
    };

    assert_eq!(expected_transaction, slp_data.transaction);

    let script = hex::decode("6a04534c500001010453454e4420550d19eb820e616a54b8a73372c4420b5a0567d8dc00f613b71c5234dc884b350800000000000003e80800000000000003e90800000000000003ea").unwrap();
    let token_id = hex::decode("550d19eb820e616a54b8a73372c4420b5a0567d8dc00f613b71c5234dc884b35").unwrap();

    let slp_data = parse_slp_script(&script).unwrap();
    assert_eq!(slp_data.lokad_id, "SLP\0");
    let expected_transaction = SlpTransaction::Send {
        token_id,
        amounts: vec![1000, 1001, 1002],
    };
    assert_eq!(expected_transaction, slp_data.transaction);
}
