set(TTS_HAVE_CUFFT ON)
set(TTS_HAVE_RELATIVE_ATTENTION ON)
set(TTS_HAVE_PRONUNCIATION ON)

add_library(tts_text OBJECT
  src/text/frontend.cpp
  src/text/language_router.cpp
  src/text/g2pw.cpp
  src/text/english.cpp
  src/text/cantonese.cpp
  src/text/morphology.cpp
  src/text/korean.cpp
  src/text/japanese.cpp
  src/text/language_detector.cpp
  src/text/budoux.cpp
  src/text/segmentation.cpp
  src/text/jieba.cpp
  src/text/pinyin.cpp
  src/text/tone_sandhi.cpp
  src/text/normalization.cpp
  src/text/phonemizer.cpp
  src/text/regex.cpp
)
set_target_properties(tts_text PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(tts_text PRIVATE include src "${PROJECT_SOURCE_DIR}")
target_link_libraries(tts_text PRIVATE nlohmann_json::nlohmann_json tts_cppjieba tts_fasttext openjtalk tts_mecab)
target_sources(tts PRIVATE
  src/ops/kernels/sequence/sequence_conditioning.cu
  src/layers/transformer.cpp
  src/io/weights.cpp
  src/execution/model_factories/gpt_sovits.cpp
  src/ops/cuda/relative_attention.cpp
  src/ops/cuda/fft.cpp
  src/ops/kernels/cpu/recurrent.cpp
  src/models/g2p_en/model.cpp
  src/models/bert/model.cpp
  src/layers/spectrogram.cpp
  src/models/gpt_sovits/model.cpp
  src/models/g2pw/model.cpp
  src/models/gpt_sovits/vocoder.cpp
  src/models/gpt_sovits/dit.cpp
  src/models/gpt_sovits/cfm.cpp
  src/models/hubert/model.cpp
  src/models/gpt_sovits/semantic.cpp
  src/layers/transformer/relative.cpp
  src/layers/audio/style_encoder.cpp
  src/layers/audio/wavenet.cpp
  src/models/gpt_sovits/acoustic.cpp
  src/models/gpt_sovits/reference_encoder.cpp
  src/inputs/phoneme_inputs.cpp
  src/ops/kernels/attention/relative_attention.cu
  src/ops/kernels/audio/spectral_kernels.cu
  src/ops/kernels/text/pronunciation.cu
)
target_link_libraries(tts PRIVATE tts_text CUDA::cufft)
