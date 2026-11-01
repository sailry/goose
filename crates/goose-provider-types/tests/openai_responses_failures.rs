use futures::StreamExt;
use goose_provider_types::errors::ProviderError;
use goose_provider_types::formats::openai_responses::{
    responses_api_to_streaming_message, ResponsesStreamEvent,
};
use serde_json::{json, Value};

fn failed(error: Value) -> Value {
    json!({
        "type": "response.failed",
        "sequence_number": 0,
        "response": {
            "id": "resp_failure",
            "status": "failed",
            "output": [],
            "error": error,
        },
    })
}

async fn first_error(frame: Value) -> ProviderError {
    let stream =
        responses_api_to_streaming_message(futures::stream::iter([Ok(format!("data: {frame}"))]));
    futures::pin_mut!(stream);
    let error = stream.next().await.unwrap().unwrap_err();
    assert!(stream.next().await.is_none());
    error.downcast::<ProviderError>().unwrap()
}

#[tokio::test]
async fn parses_nested_upstream_failure_without_full_metadata() {
    let frame = failed(json!({
        "code": "upstream_bad_gateway",
        "message": "Upstream connection closed",
        "request_id": "req_gateway",
        "private_gateway_payload": "must-not-escape",
    }));
    assert!(matches!(
        serde_json::from_value::<ResponsesStreamEvent>(frame.clone()).unwrap(),
        ResponsesStreamEvent::ResponseFailed {
            sequence_number: 0,
            ..
        }
    ));
    let error = first_error(frame).await;
    let ProviderError::ServerError(details) = error else {
        panic!("wrong failure type: {error}")
    };
    assert!(details.contains("upstream_bad_gateway"));
    assert!(details.contains("Upstream connection closed"));
    assert!(details.contains("req_gateway"));
    assert!(details.contains("resp_failure"));
    assert!(!details.contains("must-not-escape"));
    assert!(!details.contains("sequence_number"));
}

#[tokio::test]
async fn preserves_error_categories() {
    for (code, expected) in [
        ("invalid_api_key", "auth"),
        ("rate_limit_exceeded", "rate_limit"),
        ("context_length_exceeded", "context_length"),
        ("insufficient_quota", "credits_exhausted"),
        ("invalid_request_error", "request"),
        ("unknown_permanent_failure", "request"),
    ] {
        let error = first_error(failed(json!({"code": code, "message": "Specific failure"}))).await;
        assert_eq!(error.telemetry_type(), expected, "{code}");
        assert!(error.to_string().contains("Specific failure"));
    }
    let error = first_error(failed(
        json!({"code": "invalid_api_key", "type": "server_error", "message": "Bad key"}),
    ))
    .await;
    assert!(matches!(error, ProviderError::Authentication(_)));
}

#[tokio::test]
async fn classifies_flat_and_nested_error_events() {
    for frame in [
        json!({"type":"error", "code":"server_error", "message":"Service failed", "request_id":"req_error"}),
        json!({"type":"error", "error":{"code":"server_error", "message":"Service failed", "request_id":"req_error"}}),
    ] {
        let error = first_error(frame).await;
        assert!(matches!(&error, ProviderError::ServerError(_)));
        assert!(error.to_string().contains("Service failed"));
        assert!(error.to_string().contains("req_error"));
    }
}

#[tokio::test]
async fn keeps_complete_diagnostics_for_downstream_redaction() {
    let message = format!("{}credential-crossing-former-cutoff", "x".repeat(490));
    let error = first_error(failed(
        json!({"code":"upstream_bad_gateway", "message":message}),
    ))
    .await;
    assert!(error.to_string().contains(&message));
}

#[tokio::test]
async fn rejects_malformed_known_frames_without_payload_dumps() {
    let mut frame = failed(json!({"code":"upstream_bad_gateway", "message":"sensitive-payload"}));
    frame.as_object_mut().unwrap().remove("sequence_number");
    let error = first_error(frame).await;
    assert!(
        matches!(&error, ProviderError::NetworkError(details) if details.starts_with("Stream decode error:"))
    );
    assert!(error.to_string().contains("sequence_number"));
    assert!(!error.to_string().contains("sensitive-payload"));
}

#[tokio::test]
async fn treats_eof_before_completion_as_network_failure() {
    for lines in [Vec::new(), vec!["data: [DONE]".to_string()]] {
        let stream =
            responses_api_to_streaming_message(futures::stream::iter(lines.into_iter().map(Ok)));
        futures::pin_mut!(stream);
        let error = stream
            .next()
            .await
            .unwrap()
            .unwrap_err()
            .downcast::<ProviderError>()
            .unwrap();
        assert!(
            matches!(error, ProviderError::NetworkError(details) if !details.starts_with("Stream decode error:"))
        );
    }
}

#[tokio::test]
async fn preserves_partial_content_before_premature_eof_error() {
    let frame = json!({"type":"response.output_text.delta", "sequence_number":1, "item_id":"msg_1", "output_index":0, "content_index":0, "delta":"Partial"});
    let stream =
        responses_api_to_streaming_message(futures::stream::iter([Ok(format!("data: {frame}"))]));
    futures::pin_mut!(stream);
    let (message, usage) = stream.next().await.unwrap().unwrap();
    assert_eq!(message.unwrap().as_concat_text(), "Partial");
    assert!(usage.is_none());
    let error = stream
        .next()
        .await
        .unwrap()
        .unwrap_err()
        .downcast::<ProviderError>()
        .unwrap();
    assert!(matches!(error, ProviderError::NetworkError(_)));
    assert!(stream.next().await.is_none());
}

#[tokio::test]
async fn accepts_completed_and_incomplete_terminal_events() {
    for status in ["completed", "incomplete"] {
        let frame = json!({
            "type":format!("response.{status}"), "sequence_number":1,
            "response":{"id":"resp_done", "object":"response", "status":status, "model":"fixture", "output":[]}
        });
        let stream = responses_api_to_streaming_message(futures::stream::iter([Ok(format!(
            "data: {frame}"
        ))]));
        futures::pin_mut!(stream);
        let (_, usage) = stream.next().await.unwrap().unwrap();
        assert_eq!(usage.unwrap().response_id.as_deref(), Some("resp_done"));
        assert!(stream.next().await.is_none());
    }
}
