use super::command::{FromInstanceCommand, InitError, ToInstanceCommand, WS_SERVER_IP, WS_SERVER_PORT};
use common::custom_futures::{FutureTimerExt, Timeout};
use common::executor::Timer;
use common::for_tests::MarketMakerIt;
use common::log::{debug, error, info};
use common::wasm_ws::{ws_transport, WsIncomingReceiver, WsOutgoingSender};
use common::{new_uuid, panic_w, WasmUnwrapExt};
use futures::StreamExt;
use serde_json::{self as json, Value as Json};
use uuid::Uuid;
use wasm_bindgen_test::*;
use wasm_timer::TryFutureExt;
use web_sys::console;

wasm_bindgen_test_configure!(run_in_browser);

async fn receive_cmd(ws_incoming: &mut WsIncomingReceiver) -> Option<ToInstanceCommand> {
    match ws_incoming.next().await {
        Some(Ok(value)) => Some(
            json::from_value(value.clone()).expect_w(&format!("Error deserializing incoming command: {:?}", value)),
        ),
        Some(Err(e)) => {
            panic_w(&format!("Error on read next command: {:?}", e));
            unreachable!()
        },
        None => None,
    }
}

async fn send_cmd(ws_outgoing: &mut WsOutgoingSender, outgoing: FromInstanceCommand) {
    let value =
        json::to_value(outgoing.clone()).expect_w(&format!("Error serializing outgoing command: {:?}", outgoing));
    ws_outgoing.send(value).await.expect_w("Error on send command");
}

fn initialize_mm2(conf: Json) -> Result<MarketMakerIt, InitError> {
    let rpc_password = conf["rpc_password"].as_str().ok_or(InitError::NoPassInConf)?.to_owned();
    MarketMakerIt::start(conf, rpc_password, None).map_err(InitError::Other)
}

async fn run_instance(instance_id: Uuid) {
    let url = format!("ws://{}:{}", WS_SERVER_IP, WS_SERVER_PORT);
    let (mut ws_outgoing, mut ws_incoming) = ws_transport(0, &url)
        .timeout_secs(5.)
        .await
        .expect_w("Timeout connecting to the server")
        .expect_w("Error connecting to the server");

    send_cmd(&mut ws_outgoing, FromInstanceCommand::OnReady { instance_id }).await;

    let conf = match receive_cmd(&mut ws_incoming).await {
        Some(ToInstanceCommand::InitWithConf { conf }) => conf,
        Some(cmd) => {
            panic_w(&format!("Expected 'InitWithConf', found: {:?}", cmd));
            unreachable!()
        },
        None => return,
    };

    let mut mm_instance = match initialize_mm2(conf) {
        Ok(mm) => {
            send_cmd(&mut ws_outgoing, FromInstanceCommand::InitResult { result: Ok(()) }).await;
            mm
        },
        Err(e) => {
            send_cmd(&mut ws_outgoing, FromInstanceCommand::InitResult { result: Err(e) }).await;
            debug!("Allow a browser to deliver the 'InitResult'");
            Timer::sleep(1.).await;
            return;
        },
    };

    while let Some(command) = receive_cmd(&mut ws_incoming).await {
        let response = match command {
            ToInstanceCommand::Rpc { payload } => match mm_instance.rpc(payload).await {
                Ok((status_code, payload, _headers)) => FromInstanceCommand::RpcResponse {
                    status_code: status_code.as_u16(),
                    payload,
                },
                Err(error) => FromInstanceCommand::RpcFailed { error },
            },
            ToInstanceCommand::WaitForLog { timeout_sec, pattern } => {
                let result = mm_instance
                    .wait_for_log(timeout_sec, |log| log.contains(&pattern))
                    .await;
                FromInstanceCommand::LogResult { result }
            },
            cmd => {
                error!("Unexpected command: {:?}", cmd);
                continue;
            },
        };
        send_cmd(&mut ws_outgoing, response).await;
    }
}

#[wasm_bindgen_test]
async fn run_mm_instance() {
    let instance_id = new_uuid();
    info!("===== WASM MM2 Instance {} ====", instance_id);
    run_instance(instance_id).await;
    info!("===== WASM MM2 Instance {} stopped ====", instance_id);
}
