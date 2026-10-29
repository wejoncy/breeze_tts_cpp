use std::ffi::{c_char, c_void, CStr};
use std::panic::{catch_unwind, AssertUnwindSafe};
use tokenizers::Tokenizer;

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
        Tokenizer::from_file(path).map_err(|e| e.to_string())
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
    out: *mut *mut u32,
    len: *mut usize,
    err: *mut c_char,
    cap: usize,
) -> i32 {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let text = CStr::from_ptr(text).to_str().map_err(|e| e.to_string())?;
        let encoding = (&*(t as *const Tokenizer))
            .encode(text, bos)
            .map_err(|e| e.to_string())?;
        Ok::<_, String>(encoding.get_ids().to_vec().into_boxed_slice())
    }));
    match result {
        Ok(Ok(ids)) => {
            *len = ids.len();
            *out = Box::into_raw(ids) as *mut u32;
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
pub unsafe extern "C" fn hf_tokens_free(ids: *mut u32, len: usize) {
    if !ids.is_null() {
        drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(ids, len)));
    }
}
