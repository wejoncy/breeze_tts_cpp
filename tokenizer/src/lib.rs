use std::ffi::{c_char, c_void, CStr};
use std::panic::{catch_unwind, AssertUnwindSafe};
use tokenizers::Tokenizer;

#[repr(C)]
pub struct Token {
    id: u32,
    start: usize,
    end: usize,
}

unsafe fn error(dst: *mut c_char, cap: usize, msg: &str) {
    if !dst.is_null() && cap > 0 {
        let n = msg.len().min(cap - 1);
        std::ptr::copy_nonoverlapping(msg.as_ptr(), dst.cast(), n);
        *dst.add(n) = 0;
    }
}
#[no_mangle]
pub unsafe extern "C" fn hf_tokenizer_new(
    path: *const c_char,
    err: *mut c_char,
    cap: usize,
) -> *mut c_void {
    let result = catch_unwind(|| {
        let path = CStr::from_ptr(path).to_str().map_err(|e| e.to_string())?;
        let tokenizer = Tokenizer::from_file(path).map_err(|e| e.to_string())?;
        // Materialize lazy regex/BPE resources while loading, before accepting requests.
        tokenizer.encode("Warmup.", true).map_err(|e| e.to_string())?;
        Ok::<_, String>(tokenizer)
    });
    match result {
        Ok(Ok(t)) => Box::into_raw(Box::new(t)).cast(),
        Ok(Err(e)) => {
            error(err, cap, &e);
            std::ptr::null_mut()
        }
        Err(_) => {
            error(err, cap, "tokenizer panic");
            std::ptr::null_mut()
        }
    }
}
#[no_mangle]
pub unsafe extern "C" fn hf_tokenizer_free(t: *mut c_void) {
    if !t.is_null() {
        drop(Box::from_raw(t as *mut Tokenizer));
    }
}
#[no_mangle]
pub unsafe extern "C" fn hf_tokenizer_encode(
    t: *mut c_void,
    text: *const c_char,
    bos: bool,
    out: *mut *mut Token,
    len: *mut usize,
    err: *mut c_char,
    cap: usize,
) -> i32 {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let text = CStr::from_ptr(text).to_str().map_err(|e| e.to_string())?;
        let encoding = (&*(t as *const Tokenizer))
            .encode(text, bos)
            .map_err(|e| e.to_string())?;
        Ok::<_, String>(encoding.get_ids().iter().zip(encoding.get_offsets()).map(
            |(&id, &(start, end))| Token { id, start, end }
        ).collect::<Vec<_>>().into_boxed_slice())
    }));
    match result {
        Ok(Ok(ids)) => {
            *len = ids.len();
            *out = Box::into_raw(ids) as *mut Token;
            0
        }
        Ok(Err(e)) => {
            error(err, cap, &e);
            -1
        }
        Err(_) => {
            error(err, cap, "tokenizer panic");
            -1
        }
    }
}
#[no_mangle]
pub unsafe extern "C" fn hf_tokens_free(ids: *mut Token, len: usize) {
    if !ids.is_null() {
        drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(ids, len)));
    }
}

use tokenizers::utils::SysRegex;
#[repr(C)]
pub struct MatchSpan { start: usize, end: usize }

#[no_mangle]
pub unsafe extern "C" fn hf_regex_new(pattern: *const c_char, size: usize, err: *mut c_char, cap: usize) -> *mut c_void {
    let result = catch_unwind(|| {
        let pattern = std::str::from_utf8(std::slice::from_raw_parts(pattern.cast(), size)).map_err(|e| e.to_string())?;
        SysRegex::new(pattern).map_err(|e| e.to_string())
    });
    match result {
        Ok(Ok(regex)) => Box::into_raw(Box::new(regex)).cast(),
        other => {
            let message = match other { Ok(Err(e)) => e, _ => "regex panic".into() };
            error(err, cap, &message);
            std::ptr::null_mut()
        }
    }
}
#[no_mangle]
pub unsafe extern "C" fn hf_regex_free(regex: *mut c_void) {
    if !regex.is_null() { drop(Box::from_raw(regex as *mut SysRegex)); }
}
#[no_mangle]
pub unsafe extern "C" fn hf_regex_find(regex: *const c_void, text: *const c_char, size: usize,
    out: *mut *mut MatchSpan, len: *mut usize, err: *mut c_char, cap: usize) -> i32 {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let text = std::str::from_utf8(std::slice::from_raw_parts(text.cast(), size)).map_err(|e| e.to_string())?;
        Ok::<_, String>((&*(regex as *const SysRegex)).find_iter(text)
            .map(|(start, end)| MatchSpan { start, end }).collect::<Vec<_>>().into_boxed_slice())
    }));
    match result {
        Ok(Ok(spans)) => { *len = spans.len(); *out = Box::into_raw(spans) as *mut MatchSpan; 0 }
        other => {
            let message = match other { Ok(Err(e)) => e, _ => "regex panic".into() };
            error(err, cap, &message);
            -1
        }
    }
}
#[no_mangle]
pub unsafe extern "C" fn hf_regex_spans_free(spans: *mut MatchSpan, len: usize) {
    if !spans.is_null() { drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(spans, len))); }
}
