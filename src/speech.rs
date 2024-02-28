use derive_build::Build;
use serde::Serialize;

use crate::Client;

#[derive(Debug, Serialize, Copy, Clone)]
pub struct Model<'a>(pub &'a str);

impl Model<'static> {
    pub const TTS_1: Self = Self("tts-1");
    pub const TTS_1_HD: Self = Self("tts-1-hd");
}

impl Default for Model<'static> {
    fn default() -> Self {
        Self::TTS_1_HD
    }
}

#[derive(Debug, Serialize, Copy, Clone)]
pub struct Voice<'a>(pub &'a str);

impl Voice<'static> {
    pub const ALLOY: Self = Self("alloy");
    pub const ECHO: Self = Self("echo");
    pub const FABLE: Self = Self("fable");
    pub const NOVA: Self = Self("nova");
    pub const ONYX: Self = Self("onyx");
    pub const SHIMMER: Self = Self("shimmer");
}

impl Default for Voice<'static> {
    fn default() -> Self {
        Self::ALLOY
    }
}

#[derive(Build, Serialize, Copy, Clone)]
pub struct Speech<'a> {
    #[required]
    input: &'a str,

    #[required]
    #[serde(skip)]
    client: &'a Client,

    model: Model<'a>,
    voice: Voice<'a>,

    #[serde(skip_serializing_if = "Option::is_none")]
    speed: Option<f64>,
}

impl<'a> Speech<'a> {
    pub async fn send(self) -> anyhow::Result<reqwest::Response> {
        let response = self
            .client
            .client
            .post("https://api.openai.com/v1/audio/speech")
            .json(&self)
            .send()
            .await?
            .error_for_status()?;

        Ok(response)
    }
}

impl Client {
    pub fn speech<'a>(&'a self, speech: &'a str) -> Speech<'a> {
        Speech::new(speech, self)
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;

    #[tokio::test]
    async fn test_speech() -> anyhow::Result<()> {
        let client = Client::simple()?;
        let res = client
            .speech("hello there")
            .model(Model::TTS_1_HD)
            .send()
            .await?;

        // save to file test.mp3
        let mut file = std::fs::File::create("test.mp3")?;
        let bytes = res.bytes().await?;
        file.write_all(&bytes)?;

        Ok(())
    }
}
