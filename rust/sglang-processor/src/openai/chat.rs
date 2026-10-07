//! `/v1/chat/completions`, as `serving_chat.py` serves it, for requests without
//! tools. The host renders the prompt; this builds everything around it.

use std::collections::HashMap;
use std::sync::Arc;

use serde::Deserialize;
use serde_json::{Map, Value, json};

use super::dsml::{DsmlDetector, ToolCallItem};
use super::pieces::byte_level_byte;
use super::wire::{
    Reply, engine_error_reply, error_reply, http_status_name, malformed_output_reply, meta_u64,
    py_strip, python_json, stream_error_event, usage,
};
use super::{OpenAiHeaders, OpenAiSettings, OpenAiTokenizer, Unsupported};
use crate::{ThinkDetector, ThinkOptions};

/// Facts about the served model that Python's chat layer reads.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct ChatModel {
    /// `get_model().context_length`.
    pub context_length: Option<u64>,
    /// `model_config.get_default_sampling_params()` under `--sampling-defaults model`.
    pub generation_config: Map<String, Value>,
}

/// `ChatCompletionRequest` with Pydantic's defaults, for the fields that reach
/// `GenerateReqInput` or the response. A body this rejects goes to the engine.
#[derive(Deserialize)]
struct ChatRequest {
    messages: Vec<Value>,
    #[serde(default = "default_model")]
    model: String,
    #[serde(default)]
    frequency_penalty: f64,
    logit_bias: Option<Map<String, Value>>,
    #[serde(default)]
    logprobs: bool,
    top_logprobs: Option<i64>,
    max_tokens: Option<i64>,
    max_completion_tokens: Option<i64>,
    #[serde(default = "one")]
    n: i64,
    #[serde(default)]
    presence_penalty: f64,
    response_format: Option<ResponseFormat>,
    seed: Option<i64>,
    stop: Option<OneOrList<String>>,
    #[serde(default)]
    stream: bool,
    stream_options: Option<StreamOptions>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    #[allow(dead_code)]
    user: Option<String>,
    top_k: Option<i64>,
    min_p: Option<f64>,
    #[serde(default)]
    min_tokens: i64,
    regex: Option<String>,
    ebnf: Option<String>,
    repetition_penalty: Option<f64>,
    stop_token_ids: Option<Vec<i64>>,
    stop_regex: Option<OneOrList<String>>,
    #[serde(default)]
    no_stop_trim: bool,
    #[serde(default)]
    ignore_eos: bool,
    #[serde(default)]
    continue_final_message: bool,
    #[serde(default = "yes")]
    skip_special_tokens: bool,
    lora_path: Option<OneOrList<Option<String>>>,
    session_id: Option<String>,
    #[serde(default = "yes")]
    separate_reasoning: bool,
    #[serde(default = "yes")]
    stream_reasoning: bool,
    chat_template_kwargs: Option<Map<String, Value>>,
    custom_logit_processor: Option<OneOrList<Option<String>>>,
    custom_params: Option<Map<String, Value>>,
    images_config: Option<Map<String, Value>>,
    rid: Option<OneOrList<String>>,
    extra_key: Option<OneOrList<String>>,
    cache_salt: Option<OneOrList<String>>,
    priority: Option<i64>,
    data_parallel_rank: Option<i64>,
    bootstrap_host: Option<OneOrList<String>>,
    bootstrap_port: Option<OneOrList<Option<i64>>>,
    bootstrap_room: Option<OneOrList<i64>>,
    routed_dp_rank: Option<i64>,
    disagg_prefill_dp_rank: Option<i64>,
}

fn default_model() -> String {
    "default".into()
}
fn one() -> i64 {
    1
}
fn yes() -> bool {
    true
}

#[derive(Deserialize, serde::Serialize)]
#[serde(untagged)]
enum OneOrList<T> {
    One(T),
    List(Vec<T>),
}

#[derive(Deserialize)]
struct StreamOptions {
    include_usage: Option<bool>,
    continuous_usage_stats: Option<bool>,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ResponseFormat {
    Text,
    JsonObject,
    JsonSchema { json_schema: JsonSchemaFormat },
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct JsonSchemaFormat {
    name: String,
    description: Option<String>,
    schema: Option<Map<String, Value>>,
    strict: Option<bool>,
}

/// Fields that change behavior this module does not reproduce yet.
const UNSUPPORTED_FIELDS: &[&str] = &[
    "input_ids",
    "return_hidden_states",
    "return_routed_experts",
    "return_cached_tokens_details",
    "return_spec_tokens_details",
    "return_prompt_token_ids",
    "return_token_ids",
    "return_meta_info",
    "return_input_ids_in_sglext",
    "return_output_ids_in_sglext",
    "return_sampling_mask",
    "sampling_logprobs_mode",
    "video_config",
    "max_dynamic_patch",
    "min_dynamic_patch",
    "use_audio_in_video",
];

const TASKS: &[&str] = &[
    "action",
    "query",
    "authority",
    "domain",
    "title",
    "read_url",
];

const EFFORT_TIERS: &[&str] = &["none", "minimal", "low", "medium", "high", "xhigh", "max"];

/// Lower a chat body into a `/generate` body, as
/// `OpenAIServingChat._convert_to_internal_request` does. `render` returns the
/// prompt's token ids for the body as sent, or `None` when it cannot.
pub fn lower_chat(
    body: &[u8],
    headers: &OpenAiHeaders<'_>,
    settings: &OpenAiSettings,
    model: &ChatModel,
    render: impl FnOnce(&Value) -> Option<Vec<u32>>,
    tokenizer: Option<Arc<dyn OpenAiTokenizer>>,
) -> Result<(Value, ChatResponder), Unsupported> {
    let original: Value =
        serde_json::from_slice(body).map_err(|_| Unsupported("unparsed_request"))?;
    let Value::Object(mut raw) = original.clone() else {
        return Err(Unsupported("unparsed_request"));
    };
    if UNSUPPORTED_FIELDS
        .iter()
        .any(|field| raw.get(*field).is_some_and(|v| v != false))
        || raw
            .get("messages")
            .and_then(Value::as_array)
            .is_some_and(|m| m.iter().any(has_tools_or_media))
    {
        return Err(Unsupported("unsupported_chat_feature"));
    }
    let labels = super::completions::custom_labels(headers, settings);
    if !super::completions::custom_labels_are_strings(&labels) {
        return Err(Unsupported("invalid_request"));
    }
    if settings.return_input_ids || settings.return_output_ids || headers.sglext_ids {
        return Err(Unsupported("sglext_output"));
    }
    let parser = settings.reasoning_parser.as_deref();
    if parser.is_some_and(|parser| !ThinkDetector::supports(parser)) {
        return Err(Unsupported("reasoning_parser"));
    }
    let tools = request_tools(&raw)?;
    let tool_parser = settings
        .tool_call_parser
        .as_deref()
        .filter(|_| tools.is_some());
    let tool_names = match tool_parser {
        Some(parser) => {
            let names = tools.clone().unwrap_or_default();
            DsmlDetector::for_parser(parser, names.clone()).ok_or(Unsupported("tool_parser"))?;
            Some((parser.to_owned(), names))
        }
        None => None,
    };
    // Fields the typed request below does not read, typed as Pydantic types them.
    let task = raw.get("task").filter(|t| !t.is_null());
    if task.is_some_and(|t| !TASKS.iter().any(|task| t == task))
        || raw
            .get("session_params")
            .is_some_and(|p| !p.is_object() && !p.is_null())
    {
        return Err(Unsupported("invalid_request"));
    }
    let thinking_kwarg = normalize_reasoning_inputs(&mut raw)?;
    if raw
        .get("reasoning_effort")
        .is_some_and(|e| !valid_effort(e))
    {
        return Err(Unsupported("invalid_request"));
    }
    let mut request: ChatRequest =
        serde_json::from_value(Value::Object(raw)).map_err(|_| Unsupported("unparsed_request"))?;
    validate(&request, settings, model)?;
    if request.logprobs && tokenizer.is_none() {
        return Err(Unsupported("no_tokenizer"));
    }
    let prompt_ids = render(&original).ok_or(Unsupported("render"))?;

    // `_process_messages`: the server's default kwargs fill in the request's.
    let mut kwargs = request.chat_template_kwargs.take().unwrap_or_default();
    if let Some(thinking) = thinking_kwarg {
        kwargs.entry("thinking").or_insert(thinking.clone());
        kwargs.entry("enable_thinking").or_insert(thinking);
    }
    kwargs.remove("reasoning_effort");
    for (key, value) in settings.default_chat_template_kwargs.iter().flatten() {
        kwargs.entry(key.clone()).or_insert_with(|| value.clone());
    }
    let thinking = parser.is_some() && kwargs.get("thinking") == Some(&Value::Bool(true));
    // `_process_messages`: tool output keeps the parser's special tokens.
    if tools.is_some() {
        request.skip_special_tokens = false;
    }
    let previous_content = match request.messages.last() {
        Some(last)
            if request.continue_final_message
                && last["role"]
                    .as_str()
                    .is_some_and(|r| r.eq_ignore_ascii_case("assistant")) =>
        {
            // Python's reasoning parser raises on null content.
            // Roles are compared lowercased, as Pydantic normalizes them.
            if last["content"].is_null() && parser.is_some() && request.separate_reasoning {
                return Err(Unsupported("invalid_request"));
            }
            Some(last["content"].as_str().unwrap_or_default().to_owned())
        }
        _ => None,
    };

    let lowered = generate_body(
        &request, &kwargs, model, prompt_ids, thinking, headers, settings,
    );
    let (include_usage, continuous_usage_stats) = match &request.stream_options {
        Some(options) => (
            options.include_usage.unwrap_or(false)
                || settings.stream_response_default_include_usage,
            options.continuous_usage_stats.unwrap_or(false),
        ),
        None => (settings.stream_response_default_include_usage, false),
    };
    let reasoning = parser
        .filter(|_| request.separate_reasoning)
        .map(|parser| ReasoningConfig {
            parser: parser.to_owned(),
            options: ThinkOptions {
                force_reasoning: thinking,
                stream_reasoning: request.stream_reasoning,
                previous_content,
                force_nonempty_content: kwargs.get("force_nonempty_content")
                    == Some(&Value::Bool(true)),
            },
        });
    let responder = ChatResponder {
        model: request.model,
        n: request.n as usize,
        logprobs: request.logprobs,
        reasoning,
        has_parser: parser.is_some() || settings.tool_call_parser.is_some(),
        tools: tool_names,
        include_usage,
        continuous_usage_stats,
        settings: settings.clone(),
        tokenizer,
        stream: StreamState::default(),
    };
    Ok((lowered, responder))
}

/// The names of the tools a request lets the model call, or `None` when
/// `tool_choice` turns them off. Only `auto` without strict tools is served:
/// it adds no constraint, while the others depend on the engine's xgrammar.
fn request_tools(raw: &Map<String, Value>) -> Result<Option<Vec<String>>, Unsupported> {
    let choice = raw.get("tool_choice").filter(|c| !c.is_null());
    if choice.is_some_and(|c| c != "auto" && c != "none")
        || raw
            .get("parallel_tool_calls")
            .is_some_and(|p| !p.is_boolean())
    {
        return Err(Unsupported("tool_choice"));
    }
    let tools = match raw.get("tools") {
        None | Some(Value::Null) => return Ok(None),
        Some(Value::Array(tools)) => tools,
        Some(_) => return Err(Unsupported("invalid_request")),
    };
    let mut names = Vec::with_capacity(tools.len());
    for tool in tools {
        let function = &tool["function"];
        let well_formed = tool.get("type").is_none_or(Value::is_string)
            && tool.get("defer_loading").is_none_or(Value::is_null)
            && function.get("defer_loading").is_none_or(Value::is_null)
            && function.get("strict").is_none_or(|s| s == false)
            && function
                .get("description")
                .is_none_or(|d| d.is_string() || d.is_null())
            && (function["parameters"].is_null() || schema_is_standard(&function["parameters"]));
        let (true, Some(name)) = (well_formed, function["name"].as_str()) else {
            return Err(Unsupported("tool_definition"));
        };
        names.push(name.to_owned());
    }
    let parse = !tools.is_empty() && choice.is_none_or(|c| c == "auto");
    Ok(parse.then_some(names))
}

const SCHEMA_TYPES: &[&str] = &[
    "string", "number", "integer", "boolean", "object", "array", "null",
];

/// A schema `normalize_json_schema_types` leaves as is and `check_schema`
/// accepts, judged on the keywords tool schemas use.
fn schema_is_standard(schema: &Value) -> bool {
    let Value::Object(map) = schema else {
        return schema.is_boolean();
    };
    let schemas = |v: &Value| {
        v.as_object()
            .is_some_and(|m| m.values().all(schema_is_standard))
    };
    let schema_list = |v: &Value| {
        v.as_array()
            .is_some_and(|l| !l.is_empty() && l.iter().all(schema_is_standard))
    };
    let unique_strings = |v: &Value, allowed: Option<&[&str]>| {
        v.as_array().is_some_and(|l| {
            let set: std::collections::HashSet<_> = l.iter().filter_map(Value::as_str).collect();
            set.len() == l.len() && allowed.is_none_or(|a| set.iter().all(|t| a.contains(t)))
        })
    };
    let strings = |v: &Value| v.as_array().is_some_and(|l| l.iter().all(Value::is_string));
    map.iter().all(|(key, value)| match key.as_str() {
        "type" => match value {
            Value::String(t) => SCHEMA_TYPES.contains(&t.as_str()),
            types => {
                !types.as_array().is_some_and(Vec::is_empty)
                    && unique_strings(types, Some(SCHEMA_TYPES))
            }
        },
        "properties" | "$defs" | "definitions" => schemas(value),
        "items" | "additionalProperties" | "not" | "contains" | "contentSchema" => {
            schema_is_standard(value)
        }
        "anyOf" | "oneOf" | "allOf" | "prefixItems" => schema_list(value),
        "required" => unique_strings(value, None),
        "enum" | "examples" => value.is_array(),
        "description" | "title" | "format" | "$ref" | "$comment" | "$schema" | "$id"
        | "$anchor" | "$dynamicRef" | "$dynamicAnchor" | "contentEncoding" | "contentMediaType" => {
            value.is_string()
        }
        "minimum" | "maximum" | "exclusiveMinimum" | "exclusiveMaximum" => value.is_number(),
        "multipleOf" => value.as_f64().is_some_and(|m| m > 0.0),
        "minLength" | "maxLength" | "minItems" | "maxItems" | "minProperties" | "maxProperties"
        | "minContains" | "maxContains" => value.is_u64(),
        "uniqueItems" | "readOnly" | "writeOnly" | "deprecated" => value.is_boolean(),
        "dependentRequired" => value.as_object().is_some_and(|m| m.values().all(strings)),
        // Python `re` and Rust `regex` accept different patterns; normalization
        // also walks the remaining keywords, which tool schemas rarely use.
        "pattern"
        | "patternProperties"
        | "dependentSchemas"
        | "dependencies"
        | "if"
        | "then"
        | "else"
        | "propertyNames"
        | "unevaluatedItems"
        | "unevaluatedProperties" => false,
        _ => true,
    })
}

fn truthy(value: &Value) -> bool {
    !matches!(value, Value::Null | Value::Bool(false))
        && value.as_array().is_none_or(|items| !items.is_empty())
}

fn has_tools_or_media(message: &Value) -> bool {
    let media = |part: &Value| {
        matches!(
            part["type"].as_str(),
            Some("image_url" | "video_url" | "audio_url" | "input_audio")
        )
    };
    message.get("tools").is_some_and(truthy)
        || message["content"]
            .as_array()
            .is_some_and(|parts| parts.iter().any(media))
}

fn valid_effort(effort: &Value) -> bool {
    match effort {
        Value::String(tier) => EFFORT_TIERS.contains(&tier.as_str()),
        Value::Number(n) => n.as_f64().is_some_and(|f| (0.0..=0.99).contains(&f)),
        _ => false,
    }
}

/// `ChatCompletionRequest.normalize_reasoning_inputs`: lift `reasoning` into
/// `reasoning_effort` and return the `thinking` it implies.
fn normalize_reasoning_inputs(raw: &mut Map<String, Value>) -> Result<Option<Value>, Unsupported> {
    let mut thinking = None;
    if let Some(Value::Object(reasoning)) = raw.get("reasoning").cloned() {
        let effort = match reasoning.get("effort").filter(|e| !e.is_null()) {
            Some(effort) => effort.clone(),
            None => reasoning
                .get("reasoning_effort")
                .cloned()
                .unwrap_or(Value::Null),
        };
        match &effort {
            Value::String(tier) if EFFORT_TIERS.contains(&tier.as_str()) => {
                raw.insert("reasoning_effort".into(), effort.clone());
            }
            Value::Number(n) => {
                raw.insert("reasoning_effort".into(), json!(n.as_f64()));
            }
            Value::Null => {}
            _ => return Err(Unsupported("invalid_request")),
        }
        let enabled = match reasoning.get("enabled").filter(|e| !e.is_null()) {
            Some(enabled) => Some(enabled),
            None => reasoning.get("enable"),
        };
        let enabled = match enabled {
            Some(Value::String(s)) => {
                ["1", "true", "yes", "y", "on"].contains(&s.trim().to_lowercase().as_str())
            }
            Some(Value::Bool(b)) => *b,
            Some(Value::Null) | None => false,
            Some(_) => return Err(Unsupported("invalid_request")),
        };
        if enabled {
            thinking = Some(Value::Bool(true));
        }
    }
    if let Some(effort) = raw.get("reasoning_effort").filter(|e| !e.is_null()) {
        if effort.is_boolean() {
            return Err(Unsupported("invalid_request"));
        }
        thinking = Some(Value::Bool(effort != "none"));
    }
    Ok(thinking)
}

/// `OpenAIServingChat._validate_request`, for the checks requests here can fail.
fn validate(
    request: &ChatRequest,
    settings: &OpenAiSettings,
    model: &ChatModel,
) -> Result<(), Unsupported> {
    let invalid = Err(Unsupported("invalid_request"));
    if request.messages.is_empty() || request.n < 1 {
        return invalid;
    }
    if request
        .chat_template_kwargs
        .as_ref()
        .is_some_and(|k| k.contains_key("chat_template"))
    {
        return invalid;
    }
    if request
        .logit_bias
        .as_ref()
        .is_some_and(|b| !b.values().all(Value::is_number))
    {
        return invalid;
    }
    let max_output = request
        .max_completion_tokens
        .filter(|&n| n != 0)
        .or(request.max_tokens);
    if let Some(max_output) = max_output.filter(|&n| n != 0) {
        let Some(context) = settings.context_length.or(model.context_length) else {
            return Err(Unsupported("context_length_unknown"));
        };
        if max_output > context as i64 && !settings.allow_auto_truncate {
            return invalid;
        }
    }
    if let Some(ResponseFormat::JsonSchema { json_schema }) = &request.response_format
        && json_schema.schema.is_none()
    {
        return invalid;
    }
    Ok(())
}

fn generate_body(
    request: &ChatRequest,
    kwargs: &Map<String, Value>,
    model: &ChatModel,
    prompt_ids: Vec<u32>,
    thinking: bool,
    headers: &OpenAiHeaders<'_>,
    settings: &OpenAiSettings,
) -> Value {
    // `ChatCompletionRequest.to_sampling_params`: request, then generation config, then OpenAI.
    let model_defaults = settings.sampling_defaults.as_deref().unwrap_or("model") == "model";
    let param = |value: Option<Value>, name: &str, default: Value| {
        let configured = model
            .generation_config
            .get(name)
            .filter(|v| model_defaults && !v.is_null());
        value.unwrap_or_else(|| configured.cloned().unwrap_or(default))
    };
    let logit_bias = request.logit_bias.as_ref().map(|bias| {
        bias.iter()
            .map(|(k, v)| (k.clone(), json!(v.as_f64())))
            .collect::<Map<_, _>>()
    });
    let max_new_tokens = request
        .max_completion_tokens
        .filter(|&n| n != 0)
        .or(request.max_tokens);
    let mut sampling = json!({
        "temperature": param(request.temperature.map(Value::from), "temperature", json!(1.0)),
        "max_new_tokens": max_new_tokens,
        "min_new_tokens": request.min_tokens,
        "stop": request.stop,
        "stop_token_ids": request.stop_token_ids,
        "stop_regex": request.stop_regex,
        "top_p": param(request.top_p.map(Value::from), "top_p", json!(1.0)),
        "top_k": param(request.top_k.map(Value::from), "top_k", json!(-1)),
        "min_p": param(request.min_p.map(Value::from), "min_p", json!(0.0)),
        "presence_penalty": request.presence_penalty,
        "frequency_penalty": request.frequency_penalty,
        "repetition_penalty": param(request.repetition_penalty.map(Value::from), "repetition_penalty", json!(1.0)),
        "regex": request.regex,
        "ebnf": request.ebnf,
        "n": request.n,
        "no_stop_trim": request.no_stop_trim,
        "ignore_eos": request.ignore_eos,
        "skip_special_tokens": request.skip_special_tokens,
        "logit_bias": logit_bias,
        "custom_params": request.custom_params,
        "sampling_seed": request.seed,
        "spaces_between_special_tokens": kwargs.get("spaces_between_special_tokens").cloned().unwrap_or(json!(true)),
    });
    match &request.response_format {
        Some(ResponseFormat::JsonSchema { json_schema }) => {
            let schema = Value::Object(json_schema.schema.clone().unwrap_or_default());
            sampling["json_schema"] = python_json(&schema).into();
        }
        Some(ResponseFormat::JsonObject) => {
            sampling["json_schema"] = r#"{"type": "object"}"#.into()
        }
        Some(ResponseFormat::Text) | None => {}
    }
    let lora_path = match request.model.split_once(':') {
        Some((_, adapter)) if !py_strip(adapter).is_empty() => json!(py_strip(adapter)),
        _ => json!(request.lora_path),
    };
    let routed_dp_rank = match headers
        .data_parallel_rank
        .and_then(|rank| rank.parse::<i64>().ok())
    {
        Some(rank) => Some(rank),
        None => request.routed_dp_rank.or(request.data_parallel_rank),
    };
    json!({
        "input_ids": prompt_ids,
        "image_data": null,
        "video_data": null,
        "audio_data": null,
        "sampling_params": sampling,
        "return_logprob": request.logprobs,
        "logprob_start_len": -1,
        "top_logprobs_num": request.top_logprobs.unwrap_or(0),
        "return_sampling_mask": false,
        "sampling_logprobs_mode": null,
        "stream": request.stream,
        "return_text_in_logprobs": true,
        "modalities": [],
        "lora_path": lora_path,
        "bootstrap_host": request.bootstrap_host,
        "bootstrap_port": request.bootstrap_port,
        "bootstrap_room": request.bootstrap_room,
        "routed_dp_rank": routed_dp_rank,
        "disagg_prefill_dp_rank": request.disagg_prefill_dp_rank,
        "return_hidden_states": false,
        "return_routed_experts": false,
        "routed_experts_start_len": 0,
        "rid": request.rid,
        "session_id": request.session_id,
        "extra_key": request.extra_key,
        "cache_salt": request.cache_salt,
        "require_reasoning": thinking,
        "priority": request.priority,
        "routing_key": headers.routing_key,
        "custom_labels": super::completions::custom_labels(headers, settings),
        "custom_logit_processor": request.custom_logit_processor,
        "images_config": request.images_config,
        "video_config": null,
        "image_max_dynamic_patch": null,
        "video_max_dynamic_patch": null,
        "max_dynamic_patch": null,
        "use_audio_in_video": false,
        "return_prompt_token_ids": false,
    })
}

struct ReasoningConfig {
    parser: String,
    options: ThinkOptions,
}

impl ReasoningConfig {
    fn detector(&self, stream_reasoning: bool) -> ThinkDetector {
        let options = ThinkOptions {
            stream_reasoning,
            ..self.options.clone()
        };
        ThinkDetector::for_parser(&self.parser, options).expect("checked when lowering")
    }
}

/// Builds the OpenAI chat response from the engine's `/generate` output.
pub struct ChatResponder {
    model: String,
    n: usize,
    logprobs: bool,
    reasoning: Option<ReasoningConfig>,
    /// `--reasoning-parser` or `--tool-call-parser` is set.
    has_parser: bool,
    /// The tool parser and tool names when the output is parsed for tool calls.
    tools: Option<(String, Vec<String>)>,
    include_usage: bool,
    continuous_usage_stats: bool,
    settings: OpenAiSettings,
    tokenizer: Option<Arc<dyn OpenAiTokenizer>>,
    stream: StreamState,
}

#[derive(Default)]
struct StreamState {
    started: bool,
    stopped: bool,
    failed: bool,
    last_id: Value,
    roles_sent: Vec<u64>,
    text_offsets: HashMap<u64, usize>,
    logprob_counts: HashMap<u64, usize>,
    detectors: HashMap<u64, ThinkDetector>,
    tool_detectors: HashMap<u64, DsmlDetector>,
    has_tool_calls: Vec<u64>,
    finish_reasons: Vec<(u64, Value)>,
    prompt_tokens: HashMap<u64, u64>,
    completion_tokens: HashMap<u64, u64>,
    reasoning_tokens: HashMap<u64, u64>,
    cached_tokens: HashMap<u64, u64>,
}

impl ChatResponder {
    /// `_handle_non_streaming_request` over a `/generate` response.
    pub fn unary(self, status: u16, body: &[u8]) -> Option<Reply> {
        if status != 200 {
            return engine_error_reply(status, body);
        }
        let ret = match serde_json::from_slice(body).ok()? {
            Value::Array(items) => items,
            item => vec![item],
        };
        let mut choices = Vec::with_capacity(ret.len());
        for (index, item) in ret.iter().enumerate() {
            let meta = &item["meta_info"];
            let logprobs = self.logprobs.then(|| {
                json!({"content": self.token_logprobs(&meta["output_token_logprobs"], &meta["output_top_logprobs"])})
            });
            let Some(text) = item["text"].as_str() else {
                return Some(malformed_output_reply("text"));
            };
            let mut text = text.to_owned();
            let mut reasoning = String::new();
            if let Some(config) = &self.reasoning {
                (reasoning, text) = config.detector(false).parse(&text);
            }
            // `_process_tool_calls`.
            let mut finish_reason = meta["finish_reason"].clone();
            let mut tool_calls = Value::Null;
            if let Some(detector) = self.tool_detector()
                && detector.has_tool_call(&text)
            {
                let (normal, calls) = detector.parse_non_stream(&text);
                text = normal;
                if !calls.is_empty() {
                    let calls = calls
                        .iter()
                        .enumerate()
                        .map(|(i, call)| tool_call(i as i64, call));
                    tool_calls = calls.collect();
                    if finish_reason["type"] == "stop" {
                        finish_reason["type"] = "tool_calls".into();
                        finish_reason["matched"] = Value::Null;
                    }
                }
            }
            let finish_reason = &finish_reason;
            choices.push(json!({
                "index": index,
                "message": {
                    "role": "assistant",
                    "content": text,
                    "reasoning_content": (!reasoning.is_empty()).then_some(reasoning),
                    "tool_calls": tool_calls,
                },
                "logprobs": logprobs,
                "finish_reason": finish_reason.get("type"),
                "matched_stop": finish_reason.get("matched"),
            }));
        }
        let first = &ret.first()?["meta_info"];
        let stride = |key: &str| -> u64 {
            ret.iter()
                .step_by(self.n)
                .map(|r| meta_u64(&r["meta_info"], key))
                .sum()
        };
        let all = |key: &str| -> u64 { ret.iter().map(|r| meta_u64(&r["meta_info"], key)).sum() };
        let cached = self
            .settings
            .enable_cache_report
            .then(|| stride("cached_tokens"));
        let mut metadata = json!({"weight_version": first["weight_version"]});
        if let Some(versions) = first.get("weight_versions") {
            metadata["weight_versions"] = versions.clone();
        }
        let response = json!({
            "id": first["id"],
            "object": "chat.completion",
            "created": super::completions::now(),
            "model": self.model,
            "choices": choices,
            "usage": usage(stride("prompt_tokens"), all("completion_tokens"), all("reasoning_tokens"), cached),
            "metadata": metadata,
        });
        Some(Reply {
            status: 200,
            body: response.to_string(),
        })
    }

    /// One `/generate` SSE `data:` payload, as `_generate_chat_stream` handles
    /// it. `Err` replaces the whole stream, before anything was sent.
    pub fn stream_data(&mut self, data: &[u8]) -> Result<Vec<String>, Reply> {
        if data == b"[DONE]" {
            return Ok(self.stream_end());
        }
        if self.stream.stopped || self.stream.failed {
            return Ok(Vec::new());
        }
        let Ok(content) = serde_json::from_slice::<Value>(data) else {
            return Ok(Vec::new());
        };
        if let Some(message) = content.pointer("/error/message").and_then(Value::as_str) {
            if !self.stream.started {
                return Err(error_reply(message, "BadRequestError", 400));
            }
            self.stream.failed = true;
            return Ok(vec![stream_error_event(message, "BadRequestError", 400)]);
        }
        Ok(self.stream_frame(&content))
    }

    fn stream_frame(&mut self, content: &Value) -> Vec<String> {
        let index = content.get("index").and_then(Value::as_u64).unwrap_or(0);
        let meta = &content["meta_info"];
        let state = &mut self.stream;
        state.last_id = meta["id"].clone();
        state
            .prompt_tokens
            .insert(index, meta_u64(meta, "prompt_tokens"));
        state
            .completion_tokens
            .insert(index, meta_u64(meta, "completion_tokens"));
        state
            .reasoning_tokens
            .insert(index, meta_u64(meta, "reasoning_tokens"));
        state
            .cached_tokens
            .insert(index, meta_u64(meta, "cached_tokens"));
        let finish_reason = meta["finish_reason"].clone();
        let finish_type = finish_reason["type"].as_str().map(str::to_owned);

        let mut logprobs = None;
        if self.logprobs {
            let prev = self.stream.logprob_counts.get(&index).copied().unwrap_or(0);
            let total = meta_u64(meta, "output_token_logprobs_length") as usize;
            if prev < total {
                let incremental = self.settings.incremental_streaming_output;
                let slice = |key: &str| -> Value {
                    let items = meta[key].as_array().cloned().unwrap_or_default();
                    match incremental {
                        true => Value::Array(items),
                        false => {
                            Value::Array(items.into_iter().skip(prev).take(total - prev).collect())
                        }
                    }
                };
                let content = self.token_logprobs(
                    &slice("output_token_logprobs"),
                    &slice("output_top_logprobs"),
                );
                logprobs = Some(json!({"content": content}));
            }
            self.stream.logprob_counts.insert(index, total);
        }

        let mut events = Vec::new();
        if let Some(finish_type) = &finish_type {
            // An abort carrying a status is an error; Python checks for an
            // `HTTPStatus`, which `/generate` serializes as its integer.
            if finish_type == "abort"
                && let Some(code) = finish_reason["status_code"].as_u64()
                && let Some(name) = http_status_name(code)
            {
                let message = finish_reason["message"]
                    .as_str()
                    .unwrap_or("Generation aborted.");
                self.stream.stopped = true;
                self.stream.started = true;
                return vec![stream_error_event(message, name, code as u16)];
            }
            self.stream
                .finish_reasons
                .push((index, finish_reason.clone()));
        }
        if !self.stream.roles_sent.contains(&index) {
            self.stream.roles_sent.push(index);
            self.stream.started = true;
            events.push(self.chunk(
                index,
                json!({"reasoning_content": null, "role": "assistant", "content": ""}),
                None,
                None,
            ));
        }

        // `_generate_stream_content`.
        let text = content["text"].as_str().unwrap_or_default();
        let mut delta = match self.settings.incremental_streaming_output {
            true => text.to_owned(),
            false => {
                let offset = self
                    .stream
                    .text_offsets
                    .insert(index, text.chars().count())
                    .unwrap_or(0);
                text.chars().skip(offset).collect()
            }
        };
        let mut remaining_logprobs = logprobs;
        if let Some(config) = &self.reasoning {
            let detector = self
                .stream
                .detectors
                .entry(index)
                .or_insert_with(|| config.detector(config.options.stream_reasoning));
            let (mut reasoning, mut normal) = detector.push(&delta);
            if finish_type.as_deref().is_some_and(|t| t != "abort") {
                let (end_reasoning, end_normal) = detector.finish();
                reasoning.push_str(&end_reasoning);
                normal.push_str(&end_normal);
            }
            delta = normal;
            if !reasoning.is_empty() {
                let usage = self.chunk_usage(index, meta);
                events.push(self.chunk(
                    index,
                    json!({"reasoning_content": reasoning}),
                    remaining_logprobs.take(),
                    usage,
                ));
            }
        }
        if self.tools.is_some() {
            // `_process_tool_call_stream`, whose chunks are Pydantic models.
            let flush = finish_type.as_deref().is_some_and(|t| t != "abort");
            if !self.stream.tool_detectors.contains_key(&index) {
                let detector = self.tool_detector().expect("tools are set");
                self.stream.tool_detectors.insert(index, detector);
            }
            let detector = self
                .stream
                .tool_detectors
                .get_mut(&index)
                .expect("inserted");
            let (normal, calls) = detector.parse_stream(&delta, flush);
            let usage = self.chunk_usage(index, meta).unwrap_or(Value::Null);
            let delta = |content: Value, tool_calls: Value| json!({"role": null, "content": content, "reasoning_content": null, "tool_calls": tool_calls});
            if !normal.is_empty() {
                let mut chunk =
                    self.chunk_json(index, delta(normal.into(), Value::Null), None, None);
                chunk["usage"] = usage.clone();
                events.push(format!("data: {chunk}\n\n"));
            }
            for call in calls {
                if !self.stream.has_tool_calls.contains(&index) {
                    self.stream.has_tool_calls.push(index);
                }
                let mut tool_call = tool_call(call.tool_index, &call);
                // Python streams an id and name only for a non-empty name.
                if call.name.is_empty() {
                    tool_call["id"] = Value::Null;
                    tool_call["function"]["name"] = Value::Null;
                }
                let calls = json!([tool_call]);
                let mut chunk = self.chunk_json(index, delta(Value::Null, calls), None, None);
                chunk["usage"] = usage.clone();
                events.push(format!("data: {chunk}\n\n"));
            }
        } else if !delta.is_empty() {
            let usage = self.chunk_usage(index, meta);
            events.push(self.chunk(
                index,
                json!({"reasoning_content": null, "content": delta}),
                remaining_logprobs.take(),
                usage,
            ));
        }
        if remaining_logprobs.is_some() && self.has_parser {
            let usage = self.chunk_usage(index, meta);
            events.push(self.chunk(
                index,
                json!({"reasoning_content": null}),
                remaining_logprobs,
                usage,
            ));
        }
        events
    }

    fn stream_end(&mut self) -> Vec<String> {
        let mut events = Vec::new();
        if !self.stream.failed && self.stream.started {
            for (index, reason) in std::mem::take(&mut self.stream.finish_reasons) {
                let mut chunk =
                    self.chunk_json(index, json!({"reasoning_content": null}), None, None);
                let tool_calls = self.stream.has_tool_calls.contains(&index);
                chunk["choices"][0]["finish_reason"] = match reason["type"].as_str() {
                    Some("stop") if tool_calls => "tool_calls".into(),
                    _ => reason["type"].clone(),
                };
                chunk["choices"][0]["matched_stop"] =
                    reason.get("matched").cloned().unwrap_or(Value::Null);
                events.push(format!("data: {chunk}\n\n"));
            }
            if self.include_usage {
                let state = &self.stream;
                let first_choice = |map: &HashMap<u64, u64>| -> u64 {
                    map.iter()
                        .filter(|(i, _)| *i % self.n as u64 == 0)
                        .map(|(_, v)| v)
                        .sum()
                };
                let cached = self
                    .settings
                    .enable_cache_report
                    .then(|| first_choice(&state.cached_tokens));
                let chunk = json!({
                    "id": state.last_id,
                    "object": "chat.completion.chunk",
                    "created": super::completions::now(),
                    "model": self.model,
                    "choices": [],
                    "usage": usage(
                        first_choice(&state.prompt_tokens),
                        state.completion_tokens.values().sum(),
                        state.reasoning_tokens.values().sum(),
                        cached,
                    ),
                });
                events.push(format!("data: {chunk}\n\n"));
            }
        }
        events.push("data: [DONE]\n\n".into());
        events
    }

    /// `build_sse_content`: msgspec drops a null `usage`, never a null choice field.
    fn chunk(
        &self,
        index: u64,
        delta: Value,
        logprobs: Option<Value>,
        usage: Option<Value>,
    ) -> String {
        format!(
            "data: {}\n\n",
            self.chunk_json(index, delta, logprobs, usage)
        )
    }

    fn chunk_json(
        &self,
        index: u64,
        delta: Value,
        logprobs: Option<Value>,
        usage: Option<Value>,
    ) -> Value {
        let mut chunk = json!({
            "id": self.stream.last_id,
            "object": "chat.completion.chunk",
            "created": super::completions::now(),
            "model": self.model,
            "choices": [{
                "index": index,
                "delta": delta,
                "logprobs": logprobs,
                "finish_reason": null,
                "matched_stop": null,
            }],
        });
        if let Some(usage) = usage {
            chunk["usage"] = usage;
        }
        chunk
    }

    fn tool_detector(&self) -> Option<DsmlDetector> {
        let (parser, names) = self.tools.as_ref()?;
        Some(DsmlDetector::for_parser(parser, names.clone()).expect("checked when lowering"))
    }

    fn chunk_usage(&self, index: u64, meta: &Value) -> Option<Value> {
        let state = &self.stream;
        self.continuous_usage_stats.then(|| {
            let cached = self
                .settings
                .enable_cache_report
                .then(|| meta_u64(meta, "cached_tokens"));
            usage(
                state.prompt_tokens[&index],
                state.completion_tokens[&index],
                state.reasoning_tokens[&index],
                cached,
            )
        })
    }

    /// `_build_token_logprobs_from_raw`.
    fn token_logprobs(&self, output: &Value, top: &Value) -> Vec<Value> {
        let top_rows = top.as_array();
        let mut out = Vec::new();
        for (i, triple) in output.as_array().into_iter().flatten().enumerate() {
            let text = triple[2].as_str().unwrap_or_default();
            let top_logprobs: Vec<Value> = top_rows
                .and_then(|rows| rows.get(i))
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .map(|t| {
                    let text = t[2].as_str().unwrap_or_default();
                    json!({"token": text, "bytes": self.token_bytes(&t[1], text), "logprob": t[0]})
                })
                .collect();
            out.push(json!({
                "token": text,
                "bytes": self.token_bytes(&triple[1], text),
                "logprob": triple[0],
                "top_logprobs": top_logprobs,
            }));
        }
        out
    }

    fn token_bytes(&self, id: &Value, text: &str) -> Vec<u8> {
        id.as_u64()
            .and_then(|id| self.tokenizer.as_ref()?.byte_level_piece(id as u32))
            .and_then(|piece| {
                piece
                    .chars()
                    .map(byte_level_byte)
                    .collect::<Option<Vec<u8>>>()
            })
            .filter(|bytes| !bytes.is_empty())
            .unwrap_or_else(|| text.as_bytes().to_vec())
    }
}

/// A `ToolCall` as Pydantic dumps it; `call_<24 hex>` ids as SGLang mints them.
fn tool_call(index: i64, call: &ToolCallItem) -> Value {
    let id = uuid::Uuid::new_v4().simple().to_string();
    json!({
        "id": format!("call_{}", &id[..24]),
        "index": index,
        "type": "function",
        "function": {"name": call.name, "arguments": call.parameters},
    })
}
