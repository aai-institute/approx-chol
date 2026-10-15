#![no_main]

use libfuzzer_sys::fuzz_target;

#[path = "../case.rs"]
mod case;

fuzz_target!(|data: &[u8]| {
    if let Ok(input) = postcard::from_bytes::<case::Input>(data) {
        let _rejected = input.run();
    }
});
