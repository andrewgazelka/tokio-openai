#![allow(clippy::multiple_crate_versions)]
//! API for `OpenAI`

extern crate core;

use core::fmt;
use std::fmt::{Display, Formatter};

use anyhow::Context;
use derive_build::Build;
use derive_more::Constructor;
pub use reqwest;
use schemars::JsonSchema;
use serde::{
    de,
    de::{DeserializeOwned, Visitor},
    Deserialize, Deserializer, Serialize,
};
use serde_json::Value;

use crate::util::schema;

mod speech;
mod util;
pub struct StringOrStruct(pub Option<Value>);

impl<'de> Visitor<'de> for StringOrStruct {
    type Value = Option<Value>;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("string or structure")
    }

    fn visit_str<E: de::Error>(self, value: &str) -> Result<Self::Value, E> {
        match serde_json::from_str(value) {
            Ok(val) => Ok(Some(val)),
            Err(_) => Err(E::custom("expected valid json in string format")),
        }
    }

    fn visit_map<M>(self, visitor: M) -> Result<Self::Value, M::Error>
    where
        M: de::MapAccess<'de>,
    {
        let val = Value::deserialize(de::value::MapAccessDeserializer::new(visitor))?;
        Ok(Some(val))
    }
}

fn deserialize_arguments<'de, D>(deserializer: D) -> Result<Option<Value>, D::Error>
where
    D: Deserializer<'de>,
{
    deserializer.deserialize_any(StringOrStruct(None))
}

/// Grab the `OpenAI` key from the environment
///
/// # Errors
/// Will return `Err` if the key `OPENAI_KEY` does not exist
#[inline]
pub fn openai_key() -> anyhow::Result<String> {
    std::env::var("OPENAI_API_KEY")
        .context("no OpenAI key specified. Set the variable OPENAI_API_KEY")
}

/// The `OpenAI` client
#[derive(Clone)]
pub struct Client {
    client: reqwest::Client,
}

impl Client {
    /// Create a new [`Client`] client
    pub fn new(api_key: impl Into<String>) -> anyhow::Result<Self> {
        let api_key = api_key.into();

        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            reqwest::header::AUTHORIZATION,
            reqwest::header::HeaderValue::from_str(&format!("Bearer {}", api_key)).unwrap(),
        );
        headers.insert(
            reqwest::header::CONTENT_TYPE,
            reqwest::header::HeaderValue::from_static("application/json"),
        );

        // headers too
        let client = reqwest::Client::builder()
            .default_headers(headers)
            .build()?;

        Ok(Self { client })
    }

    /// # Errors
    /// Will return `Err` if no `OpenAI` key is defined
    pub fn simple() -> anyhow::Result<Self> {
        let key = openai_key()?;
        Self::new(key)
    }
}

/// ```json
/// {"model": "text-davinci-003", "prompt": "Say this is a test", "temperature": 0, "max_tokens": 7}
/// ```
#[derive(Clone, Serialize)]
pub struct TextRequest<'a> {
    pub model: Completions,
    pub prompt: &'a str,
    pub temperature: f64,

    /// Up to 4 sequences where the API will stop generating further tokens. The returned text will
    /// not contain the stop sequence.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub stop: Vec<&'a str>,

    /// number of completions
    pub n: Option<usize>,
    pub max_tokens: usize,
}

impl Default for TextRequest<'_> {
    fn default() -> Self {
        Self {
            model: Completions::Davinci,
            prompt: "",
            temperature: 0.0,
            stop: Vec::new(),
            n: None,
            max_tokens: 1_000,
        }
    }
}

/// ```json
/// {"input": "Your text string goes here", "model":"text-embedding-ada-002"}
/// ```
#[derive(Copy, Clone, Serialize, Deserialize)]
struct EmbedRequest<'a> {
    input: &'a str,
    model: &'a str,
}

#[derive(Clone, Serialize, Deserialize)]
struct TextResponseChoice {
    text: String,
}

#[derive(Clone, Serialize, Deserialize)]
struct TextResponse {
    choices: Vec<TextResponseChoice>,
}

#[derive(Clone, Serialize, Deserialize)]
struct EmbedDataFrame {
    embedding: Vec<f32>,
}

#[derive(Clone, Serialize, Deserialize)]
struct EmbedResponse {
    data: Vec<EmbedDataFrame>,
}

impl EmbedResponse {
    fn into_embedding(self) -> Vec<f32> {
        self.data
            .into_iter()
            .next()
            .map(|e| e.embedding)
            .unwrap_or_default()
    }
}

#[derive(Serialize, Deserialize)]
struct DavinciiData<'a> {
    model: &'a str,
    prompt: &'a str,
    temperature: f64,
    max_tokens: usize,
}

/// The text model we are using. See <https://openai.com/api/pricing/>
#[derive(Copy, Clone, Default, PartialEq, Eq, Debug)]
pub enum Model {
    /// The Davinci model
    #[default]
    Davinci,
    /// The Curie model
    Curie,
    /// The Babbage model
    Babbage,
    /// The Ada model
    Ada,
}

#[derive(Serialize, Deserialize, Default, Debug, PartialEq, Eq, Copy, Clone)]
pub enum ChatModel {
    #[serde(rename = "gpt-4-turbo-preview")]
    #[default]
    Gpt4TurboPreview,

    #[serde(rename = "gpt-4-1106-preview")]
    Gpt4_1106,

    #[serde(rename = "gpt-4-0613")]
    Gpt4_0613,

    #[serde(rename = "gpt-4")]
    Gpt4,
    #[serde(rename = "gpt-3.5-turbo")]
    Turbo,

    #[serde(rename = "gpt-3.5-turbo-0301")]
    Turbo0301,
}

/// ```json
/// {"role": "system", "content": "You are a helpful assistant."},
/// {"role": "user", "content": "Who won the world series in 2020?"},
/// {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
/// {"role": "user", "content": "Where was it played?"}
/// ```
#[derive(
    Serialize,
    Deserialize,
    Debug,
    Copy,
    Clone,
    PartialOrd,
    PartialEq,
    Ord,
    Eq
)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Role {
    System,
    User,
    Assistant,
    Function,
}

#[derive(Serialize, Deserialize, Debug, Clone, Constructor)]
pub struct Msg {
    /// Usually
    pub role: Role,
    pub content: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<FunctionCall>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FunctionCall {
    pub name: String,

    #[serde(deserialize_with = "deserialize_arguments")]
    pub arguments: Option<Value>,
}

impl FunctionCall {
    pub fn into_struct<T: DeserializeOwned>(self) -> anyhow::Result<T> {
        let args = self.arguments.context("no arguments")?;
        let res = serde_json::from_value(args).context("failed to deserialize arguments")?;
        Ok(res)
    }
}

impl Default for Msg {
    fn default() -> Self {
        Self::system("")
    }
}

impl Msg {
    pub fn system(content: impl Into<String>) -> Self {
        Self::new(Role::System, Some(content.into()), None, None)
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::new(Role::User, Some(content.into()), None, None)
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new(Role::Assistant, Some(content.into()), None, None)
    }

    pub fn function(name: impl Into<String>, content: impl Serialize) -> anyhow::Result<Self> {
        let name = name.into();
        let content = serde_json::to_value(content)?;
        let content = serde_json::to_string(&content)?;

        Ok(Self::new(Role::Function, Some(content), Some(name), None))
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "snake_case")]
pub enum Delta {
    /// Usually
    Role(Role),
    Content(String),
}

impl Display for Msg {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match &self.content {
            None => f.write_str(""),
            Some(content) => f.write_str(content),
        }
    }
}

#[allow(clippy::trivially_copy_pass_by_ref)]
fn real_is_one(input: &f64) -> bool {
    (*input - 1.0).abs() < f64::EPSILON
}

#[allow(clippy::trivially_copy_pass_by_ref)]
const fn int_is_one(input: &u32) -> bool {
    *input == 1
}

#[allow(clippy::trivially_copy_pass_by_ref)]
const fn int_is_zero(input: &u32) -> bool {
    *input == 0
}

const fn empty<T>(input: &[T]) -> bool {
    input.is_empty()
}

#[derive(Build, Serialize)]
pub struct ChatRequest<'a> {
    pub model: ChatModel,
    pub messages: Vec<Msg>,

    #[serde(skip)]
    #[required]
    client: &'a Client,

    /// What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the
    /// output more random, while lower values like 0.2 will make it more focused and
    /// deterministic.
    ///
    /// OpenAI generally recommend altering this or top_p but not both.
    #[serde(skip_serializing_if = "real_is_one")]
    #[default = 1.0]
    pub temperature: f64,

    /// An alternative to sampling with temperature, called nucleus sampling, where the model
    /// considers the results of the tokens with top_p probability mass. So 0.1 means only the
    /// tokens comprising the top 10% probability mass are considered.
    ///
    /// OpenAI generally recommends altering this or temperature but not both.
    #[serde(skip_serializing_if = "real_is_one")]
    #[default = 1.0]
    pub top_p: f64,

    /// How many chat completion choices to generate for each input message.
    #[serde(skip_serializing_if = "int_is_one")]
    #[default = 1]
    pub n: u32,

    #[serde(skip_serializing_if = "empty", rename = "stop")]
    pub stop_at: Vec<String>,

    /// max tokens to generate
    ///
    /// if 0, then no limit
    #[serde(skip_serializing_if = "int_is_zero")]
    pub max_tokens: u32,

    #[serde(skip_serializing_if = "empty")]
    pub functions: Vec<Function>,
}

impl<'a> ChatRequest<'a> {
    #[must_use]
    pub fn sys_msg(mut self, msg: impl Into<String>) -> Self {
        self.messages.push(Msg::system(msg));
        self
    }

    #[must_use]
    pub fn user_msg(mut self, msg: impl Into<String>) -> Self {
        self.messages.push(Msg::user(msg));
        self
    }

    #[must_use]
    pub fn assistant_msg(mut self, msg: impl Into<String>) -> Self {
        self.messages.push(Msg::assistant(msg));
        self
    }

    pub async fn send(self) -> anyhow::Result<String> {
        let response = self.send_raw().await?;
        let choice = response
            .choices
            .into_iter()
            .next()
            .context("no choices for chat")?;

        choice.message.content.context("no content for chat")
    }

    /// # Errors
    /// Returns `Err` if there is a network error communicating to `OpenAI`
    pub async fn send_raw(self) -> anyhow::Result<ChatResponse> {
        let response: String = self
            .client
            .client
            .get("https://api.openai.com/v1/chat/completions")
            .send()
            .await
            .context("could not complete chat request")?
            .text()
            .await?;

        let response = match serde_json::from_str(&response) {
            Ok(response) => response,
            Err(e) => {
                return Err(anyhow::anyhow!(
                    "could not parse chat response {response}: {e}"
                ));
            }
        };

        Ok(response)
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ChatChoice {
    pub message: Msg,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Function {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Option<Value>,
}

impl Function {
    pub fn new<Input: JsonSchema>(name: impl Into<String>, description: impl Into<String>) -> Self {
        let schema = schema::<Input>();
        Self {
            name: name.into(),
            description: Some(description.into()),
            parameters: Some(schema),
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ChatResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub choices: Vec<ChatChoice>,
}

impl ChatResponse {
    pub fn take_first(self) -> Option<ChatChoice> {
        self.choices.into_iter().next()
    }
}

/// The text model we are using. See <https://openai.com/api/pricing/>
#[derive(Deserialize, Serialize, Copy, Clone, Default, Eq, PartialEq, Debug)]
#[allow(unused)]
pub enum Completions {
    /// The Davinci model
    #[serde(rename = "text-davinci-003")]
    #[default]
    Davinci,

    /// The Curie model
    #[serde(rename = "text-curie-001")]
    Curie,
    /// The Babbage model
    #[serde(rename = "text-babbage-001")]
    Babbage,
    /// The Ada model
    #[serde(rename = "text-ada-001")]
    Ada,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Embedding<'a>(&'a str);

#[derive(Serialize, Build)]
pub struct EmbeddingRequest<'a> {
    #[required]
    #[serde(skip)]
    client: &'a Client,

    input: &'a str,
    model: Embedding<'a>,
}

impl<'a> EmbeddingRequest<'a> {
    pub async fn send(self) -> anyhow::Result<Vec<f32>> {
        let response = self
            .client
            .client
            .post("https://api.openai.com/v1/embeddings")
            .json(&self)
            .send()
            .await?
            .error_for_status()?;

        let embed: EmbedResponse = response.json().await?;

        let result = embed.into_embedding();

        Ok(result)
    }
}

impl Embedding<'static> {
    pub const LARGE: Self = Self("text-embedding-3-large");
    pub const SMALL: Self = Self("text-embedding-3-small");
}

impl Default for Embedding<'static> {
    fn default() -> Self {
        Self::SMALL
    }
}

impl Model {
    #[allow(unused)]
    const fn text_repr(self) -> &'static str {
        match self {
            Self::Davinci => "text-davinci-003",
            Self::Curie => "text-curie-001",
            Self::Babbage => "text-babbage-001",
            Self::Ada => "text-ada-001",
        }
    }
}

impl Client {
    pub fn embed(&self) -> EmbeddingRequest {
        EmbeddingRequest::new(self)
    }

    pub fn chat(&self) -> ChatRequest {
        ChatRequest::new(self)
    }
}
