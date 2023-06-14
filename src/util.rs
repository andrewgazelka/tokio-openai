use schemars::{gen::SchemaSettings, JsonSchema};

pub fn schema<T: JsonSchema>() -> serde_json::Value {
    let mut settings = SchemaSettings::default();
    settings.meta_schema = None;

    let gen = schemars::gen::SchemaGenerator::new(settings);
    let s = gen.into_root_schema_for::<T>();

    let value = serde_json::to_value(s).unwrap();

    let serde_json::Value::Object(mut map) = value else {
        panic!("Expected a JSON object");
    };

    // remove title
    map.remove("title");

    serde_json::Value::Object(map)
}
