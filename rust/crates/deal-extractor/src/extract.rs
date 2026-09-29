//! Round-0 extraction: a single whole-corpus call producing the complete
//! deal document. There is no merge step — the agent owns the full output.

use std::collections::BTreeMap;

use api::{InputMessage, MessageRequest, OutputContentBlock, ProviderClient};
use serde_json::Value;

use crate::contract;
use crate::corpus::Corpus;
use crate::prompts;

pub struct Extractor {
    pub client: ProviderClient,
    pub model: String,
    pub max_tokens: u32,
}

impl Extractor {
    pub fn new(model: &str) -> Result<Self, String> {
        let client = ProviderClient::from_model(model).map_err(|e| format!("{e:?}"))?;
        let max_tokens = api::max_tokens_for_model(model);
        Ok(Self {
            client,
            model: model.to_string(),
            max_tokens,
        })
    }

    /// Round 0: one whole-corpus call producing the full document.
    pub async fn extract(&self, corpus: &Corpus) -> Result<Value, String> {
        let prompt = format!(
            "{}\n\n## OUTPUT CONTRACT\n{}",
            prompts::EXTRACTION_PROMPT,
            contract::render_contract(),
        );
        let text = self.call(&prompt, &corpus.render()).await?;
        parse_json(&text)
    }

    /// Send one message and return the first text block. Shared by repair.
    pub(crate) async fn call(&self, prompt: &str, corpus: &str) -> Result<String, String> {
        let user_text = format!("{prompt}\n\n--- DOCUMENTS ---\n{corpus}");
        let req = json_request(
            &self.model,
            self.max_tokens,
            vec![InputMessage::user_text(user_text)],
        );
        let r = self
            .client
            .send_message(&req)
            .await
            .map_err(|e| format!("API error: {e:?}"))?;
        response_text(&r).ok_or_else(|| "no text in response".into())
    }
}

/// Build a `MessageRequest` asking for a JSON object response.
pub(crate) fn json_request(
    model: &str,
    max_tokens: u32,
    messages: Vec<InputMessage>,
) -> MessageRequest {
    let mut extra_body = BTreeMap::new();
    extra_body.insert(
        "response_format".into(),
        serde_json::json!({ "type": "json_object" }),
    );
    MessageRequest {
        model: model.to_string(),
        max_tokens,
        messages,
        system: Some(prompts::SYSTEM.to_string()),
        tools: None,
        tool_choice: None,
        stream: false,
        temperature: None,
        top_p: None,
        frequency_penalty: None,
        presence_penalty: None,
        stop: None,
        reasoning_effort: None,
        extra_body,
    }
}

/// Extract the text from a response's first text block.
pub(crate) fn response_text(r: &api::MessageResponse) -> Option<String> {
    r.content.iter().find_map(|b| match b {
        OutputContentBlock::Text { text } => Some(text.clone()),
        _ => None,
    })
}

/// Parse LLM text as JSON, stripping markdown fences. No type coercion —
/// the schema is the contract and `validate` reports mismatches to the
/// agent for repair.
pub fn parse_json(text: &str) -> Result<Value, String> {
    let trimmed = text.trim();
    let stripped = trimmed
        .strip_prefix("```json")
        .or_else(|| trimmed.strip_prefix("```"))
        .unwrap_or(trimmed)
        .trim_end_matches("```")
        .trim();
    serde_json::from_str(stripped).map_err(|e| format!("JSON parse error: {e}\nin: {trimmed}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_strips_fences() {
        assert_eq!(parse_json("```json\n{\"a\":1}\n```").unwrap()["a"], 1);
        assert_eq!(parse_json("{\"b\":2}").unwrap()["b"], 2);
    }
}
