use std::{pin::pin, vec::IntoIter};

use anyhow::Error;
use futures_util::{
    stream,
    stream::{FlatMap, Iter},
    Stream, StreamExt,
};
use tokio_stream::wrappers::ReceiverStream;

trait Decompose<T> {
    fn decompose(self) -> Vec<T>;
}

impl Decompose<char> for String {
    fn decompose(self) -> Vec<char> {
        self.chars().collect()
    }
}

pub type CharStream<T> = FlatMap<
    T,
    Iter<IntoIter<Result<char, Error>>>,
    fn(anyhow::Result<String>) -> Iter<IntoIter<Result<char, Error>>>,
>;
pub type LineStream = ReceiverStream<anyhow::Result<String>>;

pub trait OpenAiStreamExt: Stream<Item = anyhow::Result<String>> + Sized {
    fn chars(self) -> CharStream<Self> {
        self.flat_map(|elem| {
            let result = match elem {
                Err(res) => vec![Err(res)],
                Ok(res) => res.chars().map(Ok).collect(),
            };
            stream::iter(result)
        })
    }

    /// Outputs a stream of Strings where each item is a line
    fn lines(self) -> LineStream
    where
        Self: Send + 'static,
    {
        let (tx, rx) = tokio::sync::mpsc::channel(1);

        tokio::spawn(async move {
            let mut chars = pin!(self.chars());

            let mut s = String::new();

            while let Some(char) = chars.next().await {
                let char = match char {
                    Ok(char) => char,
                    Err(e) => {
                        let _ = tx.send(Err(e)).await;
                        return;
                    }
                };

                if char == '\n' {
                    let s = core::mem::take(&mut s);
                    if tx.send(Ok(s)).await.is_err() {
                        return;
                    }
                } else {
                    s.push(char);
                }
            }
        });

        ReceiverStream::new(rx)
    }
}

impl<T> OpenAiStreamExt for T
where
    T: Stream<Item = anyhow::Result<String>>,
    T: Sized,
{
}
