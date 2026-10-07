//! Every `tests/fixtures/reasoning_parity/*.json` case splits as SGLang's
//! `ReasoningParser` does. Fixtures come from
//! `tests/scripts/generate_reasoning_parity.py`.

use std::path::Path;

use serde_json::{Value, json};
use sglang_processor::{ThinkDetector, ThinkOptions};

#[test]
fn fixtures_match_sglang() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/reasoning_parity");
    for entry in std::fs::read_dir(dir).unwrap() {
        let fixture: Value =
            serde_json::from_str(&std::fs::read_to_string(entry.unwrap().path()).unwrap()).unwrap();
        let parser = fixture["parser"].as_str().unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let chunks: Vec<&str> = case["chunks"]
                .as_array()
                .unwrap()
                .iter()
                .map(|c| c.as_str().unwrap())
                .collect();
            let detector = |stream_reasoning| {
                let options = ThinkOptions {
                    force_reasoning: case["force"] == true,
                    stream_reasoning,
                    ..Default::default()
                };
                ThinkDetector::for_parser(parser, options).unwrap()
            };
            let mut streaming = detector(case["stream"] == true);
            let mut steps: Vec<_> = chunks.iter().map(|chunk| streaming.push(chunk)).collect();
            steps.push(streaming.finish());
            let unary = detector(false).parse(&chunks.concat());
            let got = json!({"steps": steps, "unary": unary});
            let want = json!({"steps": case["steps"], "unary": case["unary"]});
            assert_eq!(got, want, "{parser} {case}");
        }
    }
}
