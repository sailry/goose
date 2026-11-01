#![cfg(not(target_arch = "wasm32"))]

use async_trait::async_trait;
use futures::{FutureExt, StreamExt};
use goose_provider_types::base::{MessageStream, Provider};
use goose_provider_types::conversation::message::Message;
use goose_provider_types::conversation::token_usage::{ProviderUsage, Usage};
use goose_provider_types::errors::ProviderError;
use goose_provider_types::model::ModelConfig;
use goose_provider_types::retry::{
    retry_operation, stream_with_retry, ProviderRetry, RetryConfig, StreamRetry,
};
use rmcp::model::{CallToolRequestParams, Tool};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::time::{advance, Instant};

type Chunk = (Option<Message>, Option<ProviderUsage>);
type Item = Result<Chunk, ProviderError>;
const INTERVAL: Duration = Duration::from_secs(60);

#[derive(Clone)]
enum Step {
    Setup(ProviderError),
    Chunks(Vec<Item>),
    Lazy(ProviderError),
    PendingSetup,
    PendingStream,
}

struct State {
    steps: Mutex<VecDeque<Step>>,
    times: Mutex<Vec<Instant>>,
    requests: AtomicUsize,
    refreshes: AtomicUsize,
    active: AtomicUsize,
    dropped: AtomicUsize,
}

#[derive(Clone)]
struct Scripted(Arc<State>);

impl Scripted {
    fn new(steps: impl IntoIterator<Item = Step>) -> Self {
        Self(Arc::new(State {
            steps: Mutex::new(steps.into_iter().collect()),
            times: Mutex::new(Vec::new()),
            requests: AtomicUsize::new(0),
            refreshes: AtomicUsize::new(0),
            active: AtomicUsize::new(0),
            dropped: AtomicUsize::new(0),
        }))
    }

    async fn request(&self, error: Option<ProviderError>) -> Result<(), ProviderError> {
        self.with_retry_config(
            || async {
                retry_operation(&RetryConfig::new(3, 1, 1.0, 1), || async {
                    self.0.requests.fetch_add(1, Ordering::SeqCst);
                    error.clone().map_or(Ok(()), Err)
                })
                .await
            },
            RetryConfig::new(3, 1, 1.0, 1),
        )
        .await
    }

    fn attempts(&self) -> usize {
        self.0.times.lock().unwrap().len()
    }
}

struct Flight(Arc<State>);

impl Flight {
    fn new(state: Arc<State>) -> Self {
        state.active.fetch_add(1, Ordering::SeqCst);
        Self(state)
    }
}

impl Drop for Flight {
    fn drop(&mut self) {
        self.0.active.fetch_sub(1, Ordering::SeqCst);
        self.0.dropped.fetch_add(1, Ordering::SeqCst);
    }
}

#[async_trait]
impl Provider for Scripted {
    fn get_name(&self) -> &str {
        "retry-fixture"
    }

    async fn stream(
        &self,
        _model: &ModelConfig,
        _system: &str,
        _messages: &[Message],
        _tools: &[Tool],
    ) -> Result<MessageStream, ProviderError> {
        self.0.times.lock().unwrap().push(Instant::now());
        let step = self
            .0
            .steps
            .lock()
            .unwrap()
            .pop_front()
            .expect("extra request");
        match step {
            Step::Setup(error) => {
                self.request(Some(error)).await?;
                unreachable!()
            }
            Step::Chunks(chunks) => {
                self.request(None).await?;
                Ok(Box::pin(futures::stream::iter(chunks)))
            }
            Step::Lazy(error) => {
                let provider = self.clone();
                Ok(Box::pin(async_stream::stream! {
                    yield provider.request(Some(error)).await.map(|()| (None, None));
                }))
            }
            Step::PendingSetup => {
                let _flight = Flight::new(self.0.clone());
                futures::future::pending().await
            }
            Step::PendingStream => {
                let state = self.0.clone();
                Ok(Box::pin(async_stream::stream! {
                    let _flight = Flight::new(state);
                    yield futures::future::pending::<Item>().await;
                }))
            }
        }
    }

    async fn refresh_credentials(&self) -> Result<(), ProviderError> {
        self.0.refreshes.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

fn stream(provider: &Scripted, policy: StreamRetry) -> MessageStream {
    stream_with_retry(
        Arc::new(provider.clone()),
        ModelConfig::new("fixture"),
        "System".into(),
        vec![Message::user().with_text("Request")],
        Vec::new(),
        policy,
    )
}

fn transient() -> ProviderError {
    ProviderError::ServerError("Upstream unavailable (provider request id: req_retry)".into())
}

fn usage(id: &str) -> ProviderUsage {
    ProviderUsage::new("fixture".into(), Usage::default()).with_request_id(id.into())
}

#[tokio::test(start_paused = true)]
async fn exhausts_five_retries_exactly_sixty_seconds_apart() {
    let _env = env_lock::lock_env([("GOOSE_PROVIDER_SKIP_BACKOFF", None::<&str>)]);
    let failures = (0..6).map(|index| {
        Step::Setup(ProviderError::RateLimitExceeded {
            details: format!("Request req_{index}"),
            retry_delay: Some(Duration::from_secs(600)),
        })
    });
    let provider = Scripted::new(failures);
    let mut output = stream(&provider, StreamRetry::fixed(5, INTERVAL));
    let started = Instant::now();
    assert!(output.next().now_or_never().is_none());
    assert_eq!(provider.attempts(), 1);

    for retry in 1..=5 {
        advance(Duration::from_secs(59)).await;
        assert!(output.next().now_or_never().is_none());
        assert_eq!(provider.attempts(), retry);
        advance(Duration::from_secs(1)).await;
        let outcome = output.next().now_or_never();
        assert_eq!(provider.attempts(), retry + 1);
        if retry < 5 {
            assert!(outcome.is_none());
        } else {
            assert_eq!(
                outcome.unwrap().unwrap().unwrap_err(),
                ProviderError::RateLimitExceeded {
                    details: "Request req_5".into(),
                    retry_delay: Some(Duration::from_secs(600)),
                }
            );
        }
    }
    assert!(output.next().await.is_none());
    assert_eq!(provider.0.requests.load(Ordering::SeqCst), 6);
    let elapsed: Vec<_> = provider
        .0
        .times
        .lock()
        .unwrap()
        .iter()
        .map(|instant| instant.duration_since(started).as_secs())
        .collect();
    assert_eq!(elapsed, [0, 60, 120, 180, 240, 300]);
}

#[tokio::test(start_paused = true)]
async fn shares_budget_across_http_and_stream_failures() {
    let _env = env_lock::lock_env([("GOOSE_PROVIDER_SKIP_BACKOFF", None::<&str>)]);
    let provider = Scripted::new([
        Step::Setup(transient()),
        Step::Lazy(transient()),
        Step::Chunks(vec![Err(ProviderError::NetworkError(
            "Connection reset".into(),
        ))]),
        Step::Setup(transient()),
        Step::Lazy(transient()),
        Step::Chunks(vec![Ok((
            Some(Message::assistant().with_text("Done")),
            None,
        ))]),
    ]);
    let output: Vec<_> = stream(&provider, StreamRetry::fixed(5, INTERVAL))
        .collect()
        .await;
    assert_eq!(output.len(), 1);
    assert_eq!(
        output[0]
            .as_ref()
            .unwrap()
            .0
            .as_ref()
            .unwrap()
            .as_concat_text(),
        "Done"
    );
    assert_eq!(provider.attempts(), 6);
    assert_eq!(provider.0.requests.load(Ordering::SeqCst), 6);
    let times = provider.0.times.lock().unwrap();
    assert!(times.windows(2).all(|pair| pair[1] - pair[0] == INTERVAL));
}

#[tokio::test(start_paused = true)]
async fn leaves_permanent_and_decode_failures_terminal() {
    let _env = env_lock::lock_env([("GOOSE_PROVIDER_SKIP_BACKOFF", None::<&str>)]);
    for error in [
        ProviderError::RequestFailed("Invalid request".into()),
        ProviderError::ContextLengthExceeded("Too long".into()),
        ProviderError::CreditsExhausted {
            details: "No credits".into(),
            top_up_url: None,
        },
        ProviderError::stream_decode_error("Malformed response frame"),
    ] {
        let provider = Scripted::new([Step::Setup(error.clone())]);
        let started = Instant::now();
        let mut output = stream(&provider, StreamRetry::fixed(5, INTERVAL));
        assert_eq!(output.next().await.unwrap().unwrap_err(), error);
        assert!(output.next().await.is_none());
        assert_eq!(provider.attempts(), 1);
        assert_eq!(Instant::now(), started);
    }
}

#[tokio::test(start_paused = true)]
async fn retains_one_auth_refresh_without_transient_replay() {
    let _env = env_lock::lock_env([("GOOSE_PROVIDER_SKIP_BACKOFF", None::<&str>)]);
    let error = ProviderError::Authentication("Invalid key (provider request id: req_auth)".into());
    let provider = Scripted::new([Step::Setup(error.clone())]);
    let mut output = stream(&provider, StreamRetry::fixed(5, INTERVAL));
    assert_eq!(output.next().await.unwrap().unwrap_err(), error);
    assert_eq!(provider.attempts(), 1);
    assert_eq!(provider.0.refreshes.load(Ordering::SeqCst), 1);
    assert_eq!(provider.0.requests.load(Ordering::SeqCst), 2);
}

#[tokio::test(start_paused = true)]
async fn never_replays_after_any_message_or_usage() {
    let _env = env_lock::lock_env([("GOOSE_PROVIDER_SKIP_BACKOFF", None::<&str>)]);
    let barriers = [
        (Some(Message::assistant()), None),
        (Some(Message::assistant().with_text("Partial")), None),
        (
            Some(
                Message::assistant()
                    .with_tool_request("call_1", Ok(CallToolRequestParams::new("tool"))),
            ),
            None,
        ),
        (None, Some(usage("req_partial"))),
    ];
    for barrier in barriers {
        let provider = Scripted::new([Step::Chunks(vec![Ok(barrier), Err(transient())])]);
        let output: Vec<_> = stream(&provider, StreamRetry::fixed(5, INTERVAL))
            .collect()
            .await;
        assert_eq!(output.len(), 2);
        assert!(output[0].is_ok());
        assert_eq!(output[1].as_ref().unwrap_err(), &transient());
        assert_eq!(provider.attempts(), 1);
    }
}

#[tokio::test(start_paused = true)]
async fn discards_failed_summary_messages_and_usage_together() {
    let _env = env_lock::lock_env([("GOOSE_PROVIDER_SKIP_BACKOFF", None::<&str>)]);
    let provider = Scripted::new([
        Step::Chunks(vec![
            Ok((Some(Message::assistant().with_text("Discarded")), None)),
            Ok((None, Some(usage("req_discarded")))),
            Err(transient()),
        ]),
        Step::Chunks(vec![
            Ok((Some(Message::assistant().with_text("Summary")), None)),
            Ok((None, Some(usage("req_summary")))),
        ]),
    ]);
    let output: Vec<_> = stream(&provider, StreamRetry::fixed(5, INTERVAL).buffered())
        .collect()
        .await;
    assert_eq!(output.len(), 2);
    assert_eq!(
        output[0]
            .as_ref()
            .unwrap()
            .0
            .as_ref()
            .unwrap()
            .as_concat_text(),
        "Summary"
    );
    assert_eq!(
        output[1]
            .as_ref()
            .unwrap()
            .1
            .as_ref()
            .unwrap()
            .request_id
            .as_deref(),
        Some("req_summary")
    );
    assert_eq!(provider.attempts(), 2);
}

#[tokio::test(start_paused = true)]
async fn dropping_cancels_pending_creation_and_stream_poll() {
    let _env = env_lock::lock_env([("GOOSE_PROVIDER_SKIP_BACKOFF", None::<&str>)]);
    for step in [Step::PendingSetup, Step::PendingStream] {
        let provider = Scripted::new([step]);
        let mut output = stream(&provider, StreamRetry::fixed(5, INTERVAL));
        assert!(output.next().now_or_never().is_none());
        assert_eq!(provider.0.active.load(Ordering::SeqCst), 1);
        drop(output);
        assert_eq!(provider.0.active.load(Ordering::SeqCst), 0);
        assert_eq!(provider.0.dropped.load(Ordering::SeqCst), 1);
        advance(Duration::from_secs(600)).await;
        assert_eq!(provider.attempts(), 1);
    }
}

#[tokio::test(start_paused = true)]
async fn dropping_during_delay_stops_future_attempts() {
    let _env = env_lock::lock_env([("GOOSE_PROVIDER_SKIP_BACKOFF", None::<&str>)]);
    let provider = Scripted::new([Step::Setup(transient())]);
    let mut output = stream(&provider, StreamRetry::fixed(5, INTERVAL));
    assert!(output.next().now_or_never().is_none());
    drop(output);
    advance(Duration::from_secs(600)).await;
    assert_eq!(provider.attempts(), 1);
}

#[tokio::test(start_paused = true)]
async fn preserves_legacy_retries_on_the_same_task() {
    let _env = env_lock::lock_env([("GOOSE_PROVIDER_SKIP_BACKOFF", None::<&str>)]);
    let provider = Scripted::new([Step::Setup(transient())]);
    let mut output = stream(&provider, StreamRetry::fixed(5, INTERVAL));
    assert!(output.next().now_or_never().is_none());
    let operation_calls = AtomicUsize::new(0);
    retry_operation(&RetryConfig::new(1, 0, 1.0, 0), || async {
        if operation_calls.fetch_add(1, Ordering::SeqCst) == 0 {
            Err(transient())
        } else {
            Ok(())
        }
    })
    .await
    .unwrap();
    let provider_calls = AtomicUsize::new(0);
    provider
        .with_retry_config(
            || async {
                if provider_calls.fetch_add(1, Ordering::SeqCst) == 0 {
                    Err(transient())
                } else {
                    Ok(())
                }
            },
            RetryConfig::new(1, 0, 1.0, 0),
        )
        .await
        .unwrap();
    assert_eq!(operation_calls.load(Ordering::SeqCst), 2);
    assert_eq!(provider_calls.load(Ordering::SeqCst), 2);
}

#[tokio::test(start_paused = true)]
async fn retains_explicit_backoff_override_for_test_callers() {
    let _env = env_lock::lock_env([("GOOSE_PROVIDER_SKIP_BACKOFF", Some("true"))]);
    let provider = Scripted::new([
        Step::Setup(transient()),
        Step::Chunks(vec![Ok((None, Some(usage("req_done"))))]),
    ]);
    let started = Instant::now();
    let output: Vec<_> = stream(&provider, StreamRetry::fixed(5, INTERVAL))
        .collect()
        .await;
    assert_eq!(output.len(), 1);
    assert!(output[0].is_ok());
    assert_eq!(provider.attempts(), 2);
    assert_eq!(Instant::now(), started);
}
