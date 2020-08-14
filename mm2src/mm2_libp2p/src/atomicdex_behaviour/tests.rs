use super::start_gossipsub;
use crate::atomicdex_behaviour::AdexBehaviorCmd;
use crate::request_response::{PeerRequest, PeerResponse};
use async_std::task::{block_on, spawn};
use futures::channel::{mpsc, oneshot};
use futures::{Future, SinkExt, StreamExt};
use secp256k1::SecretKey;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

fn spawn_boxed(fut: Box<dyn Future<Output = ()> + Send + Unpin + 'static>) { spawn(fut); }

struct Node {
    #[allow(dead_code)]
    secret: SecretKey,
    cmd_tx: mpsc::UnboundedSender<AdexBehaviorCmd>,
}

impl Node {
    fn spawn<F>(ip: String, port: u16, seednodes: Option<Vec<String>>, on_request: F) -> Node
    where
        F: Fn((PeerRequest, oneshot::Sender<PeerResponse>)) + Send + 'static,
    {
        let my_address = ip.parse().unwrap();

        let mut rng = rand::thread_rng();
        let secret = SecretKey::random(&mut rng);
        let mut priv_key = secret.serialize();

        let (cmd_tx, _gossip_event_rx, mut request_rx, _my_peer_id) =
            start_gossipsub(my_address, port, spawn_boxed, seednodes, &mut priv_key);

        // spawn a response future
        spawn(async move {
            loop {
                match request_rx.next().await {
                    Some(r) => on_request(r),
                    _ => {
                        println!("Finish response future");
                        break;
                    },
                }
            }
        });

        Node { secret, cmd_tx }
    }

    async fn send_cmd(&mut self, cmd: AdexBehaviorCmd) { self.cmd_tx.send(cmd).await.unwrap(); }

    async fn wait_peers(&mut self, number: usize) {
        loop {
            let (tx, rx) = oneshot::channel();
            self.cmd_tx
                .send(AdexBehaviorCmd::GetPeersInfo { result_tx: tx })
                .await
                .unwrap();
            match rx.await {
                Ok(map) => {
                    if map.len() >= number {
                        return;
                    }
                    async_std::task::sleep(Duration::from_millis(500)).await;
                },
                Err(e) => panic!("{}", e),
            }
        }
    }
}

#[test]
fn test_request_response_ok() {
    std::env::set_var("RUST_LOG", "debug");
    // let _ = env_logger::try_init();
    let _ = env_logger::builder().is_test(true).try_init();

    let request_received = Arc::new(AtomicBool::new(false));
    let request_received_cpy = request_received.clone();
    let _node1 = Node::spawn("127.0.0.1".into(), 57783, None, move |(request, tx)| {
        request_received_cpy.store(true, Ordering::Relaxed);

        assert_eq!(request.topic, "test:topic");
        assert_eq!(request.req, b"test request");

        assert_eq!(
            tx.send(PeerResponse::Ok {
                res: b"test response".to_vec()
            }),
            Ok(())
        );
    });
    let mut node2 = Node::spawn(
        "127.0.0.1".into(),
        57784,
        Some(vec!["/ip4/127.0.0.1/tcp/57783".into()]),
        |_| (),
    );

    block_on(async { node2.wait_peers(1).await });

    let (response_tx, response_rx) = oneshot::channel();
    block_on(async move {
        node2
            .send_cmd(AdexBehaviorCmd::SendRequest {
                req: b"test request".to_vec(),
                topic: "test:topic".into(),
                response_tx,
            })
            .await;

        let res = response_rx.await;
        assert_eq!(
            res,
            Ok(PeerResponse::Ok {
                res: b"test response".to_vec()
            })
        );
    });

    assert!(request_received.load(Ordering::Relaxed));
}
