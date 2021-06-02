use http::StatusCode;
use serde_json::{self as json, Value as Json};
use uuid::Uuid;

pub type InstanceId = Uuid;

pub const WS_SERVER_IP: &str = "127.0.0.1";
pub const WS_SERVER_PORT: u16 = 6142;

#[derive(Debug, Deserialize, Serialize)]
pub enum ToInstanceCommand {
    InitWithConf { conf: Json },
    Rpc { payload: Json },
    WaitForLog { timeout_sec: f64, pattern: String },
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum FromInstanceCommand {
    OnReady { instance_id: InstanceId },
    InitResult { result: Result<(), InitError> },
    RpcResponse { status_code: u16, payload: String },
    RpcFailed { error: String },
    LogResult { result: Result<(), String> },
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum InitError {
    NoPassInConf,
    Other(String),
}
