NVCC ?= nvcc
CXX ?= g++
AR ?= ar
ARCH ?= sm_120
NVCCFLAGS ?= -g -G
CXXFLAGS ?= -g -Wall -Wextra
CPPFLAGS += -I.

OPS_SRCS := $(wildcard ops/*.cu)
OPS_OBJS := $(patsubst ops/%.cu,build/ops/%.o,$(OPS_SRCS))
OBJS := $(OPS_OBJS)
TARGET := build/libinference_ops.a

all: $(TARGET) model resources

ops: $(TARGET)

$(TARGET): $(OBJS)
	$(AR) rcs $@ $^

build/ops/%.o: ops/%.cu ops/ops.h | build/ops
	$(NVCC) -arch=$(ARCH) $(CPPFLAGS) $(NVCCFLAGS) -std=c++17 -MMD -MP -c $< -o $@

build build/ops:
	mkdir -p $@

clean:
	rm -f $(OBJS) $(OBJS:.o=.d) $(TARGET)

.PHONY: all ops clean

-include $(OPS_OBJS:.o=.d)

# Qwen tokenizer 可以单独构建，也供完整 REPL 链接。
# 复用 Unicode 实现及 GGUF 读取库；BPE 编解码实现在本项目 tokenizer.cpp 中。
LLAMA_DIR ?= ../llama-deploy/llama.cpp
LLAMA_LIB_DIR ?= $(LLAMA_DIR)/build/bin
TOKENIZER_CPPFLAGS := -I$(LLAMA_DIR)/src -I$(LLAMA_DIR)/ggml/include
TOKENIZER_OBJS := build/tokenizer/tokenizer.o build/tokenizer/unicode.o build/tokenizer/unicode-data.o
TOKENIZER_LIB := build/libinference_tokenizer.a
# 最终可执行文件需在本静态库之后链接 $(TOKENIZER_LDLIBS)。
TOKENIZER_LDLIBS = -L$(LLAMA_LIB_DIR) -Wl,-rpath,$(abspath $(LLAMA_LIB_DIR)) -lggml-base
TOKENIZER_NVCC_LDLIBS = -L$(LLAMA_LIB_DIR) -Xlinker -rpath -Xlinker $(abspath $(LLAMA_LIB_DIR)) -lggml-base

tokenizer: $(TOKENIZER_LIB)

$(TOKENIZER_LIB): $(TOKENIZER_OBJS)
	$(AR) rcs $@ $^

build/tokenizer/tokenizer.o: src/session/tokenizer.cpp src/session/tokenizer.h $(LLAMA_DIR)/src/unicode.h $(LLAMA_DIR)/ggml/include/gguf.h | build/tokenizer
	$(CXX) $(CPPFLAGS) $(TOKENIZER_CPPFLAGS) $(CXXFLAGS) -std=c++17 -MMD -MP -c $< -o $@

# 隐藏辅助代码符号，避免同一进程链接 libllama 时 Unicode 全局表重复初始化。
build/tokenizer/%.o: $(LLAMA_DIR)/src/%.cpp Makefile | build/tokenizer
	$(CXX) $(CPPFLAGS) $(TOKENIZER_CPPFLAGS) $(CXXFLAGS) -std=c++17 -fvisibility=hidden -MMD -MP -c $< -o $@

build/tokenizer:
	mkdir -p $@

clean-tokenizer:
	rm -f $(TOKENIZER_OBJS) $(TOKENIZER_OBJS:.o=.d) $(TOKENIZER_LIB)

-include $(TOKENIZER_OBJS:.o=.d)

.PHONY: tokenizer clean-tokenizer

# Model/Layer 只编译 host 调度，最终与 ops 库链接。
MODEL_OBJS := build/model/model.o build/model/layer.o
MODEL_LIB := build/libinference_model.a

model: $(MODEL_LIB)

$(MODEL_LIB): $(MODEL_OBJS)
	$(AR) rcs $@ $^

build/model/model.o: src/model/model.cpp src/model/model.h src/model/layer/layer.h ops/ops.h | build/model
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -std=c++17 -MMD -MP -c $< -o $@

build/model/layer.o: src/model/layer/layer.cpp src/model/layer/layer.h src/model/model.h ops/ops.h | build/model
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -std=c++17 -MMD -MP -c $< -o $@

build/model:
	mkdir -p $@

clean-model:
	rm -f $(MODEL_OBJS) $(MODEL_OBJS:.o=.d) $(MODEL_LIB)

-include $(MODEL_OBJS:.o=.d)

.PHONY: model clean-model

# 原权重/RunState 资源函数。调用时机由外部所有者（如 Session）决定。
RESOURCE_OBJ := build/model/model_resources.o
RESOURCE_LIB := build/libinference_resources.a

resources: $(RESOURCE_LIB)

$(RESOURCE_LIB): $(RESOURCE_OBJ)
	$(AR) rcs $@ $^

$(RESOURCE_OBJ): src/model/model_resources.cu src/model/model_types.h | build/model
	$(NVCC) -arch=$(ARCH) $(CPPFLAGS) $(NVCCFLAGS) -std=c++17 -MMD -MP -c $< -o $@

clean-resources:
	rm -f $(RESOURCE_OBJ) $(RESOURCE_OBJ:.o=.d) $(RESOURCE_LIB)

-include $(RESOURCE_OBJ:.o=.d)

.PHONY: resources clean-resources

# 无需 GPU：替换算子入口，检查 forward 的顺序、权重、缓存位置和尺寸。
test-model: build/model/model_forward_test
	./build/model/model_forward_test

build/model/model_forward_test: tests/model_forward_test.cpp $(MODEL_LIB) ops/ops.h src/model/model_types.h | build/model
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -std=c++17 -UNDEBUG $< $(MODEL_LIB) -o $@

.PHONY: test-model

# 原仓库三项数值/生成测试的入口，只链接当前重构后的模型与算子。
CUDA_TEST_LIBS := $(MODEL_LIB) $(TARGET) $(RESOURCE_LIB)

test_kernel/test_single_forward/test_single_forward: test_kernel/test_single_forward/test_single_forward.cu $(CUDA_TEST_LIBS)
	$(NVCC) -arch=$(ARCH) $(CPPFLAGS) $(NVCCFLAGS) -std=c++17 $< $(CUDA_TEST_LIBS) -o $@

engine: test_kernel/engine.cu build/libinference_runtime.a $(CUDA_TEST_LIBS)
	$(NVCC) -arch=$(ARCH) $(CPPFLAGS) $(NVCCFLAGS) -std=c++17 $^ -o $@

test-kernel-build: test_kernel/test_single_forward/test_single_forward engine

.PHONY: test-kernel-build

# 仅用于判断 score 偏差是否由拆分引入，原融合 kernel 存在于测试文件中。
test_kernel/test_single_forward/test_score_split: test_kernel/test_single_forward/test_score_split.cu ops/attention_config.h $(CUDA_TEST_LIBS)
	$(NVCC) -arch=$(ARCH) $(CPPFLAGS) $(NVCCFLAGS) -std=c++17 $< $(CUDA_TEST_LIBS) -o $@

# 完整 REPL：普通 C++ 处理会话/生成；CUDA launch 仅在 ops。
RUNTIME_OBJS := build/runtime/generator.o build/runtime/sampler.o build/runtime/session.o
RUNTIME_LIB := build/libinference_runtime.a
REPL_OBJS := build/runtime/main.o build/runtime/terminal.o

$(RUNTIME_LIB): $(RUNTIME_OBJS)
	$(AR) rcs $@ $^

build/runtime/generator.o: src/generator/generator.cpp | build/runtime
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -std=c++17 -MMD -MP -c $< -o $@

build/runtime/sampler.o: src/generator/sampler/sampler.cpp | build/runtime
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -std=c++17 -MMD -MP -c $< -o $@

build/runtime/session.o: src/session/session.cpp | build/runtime
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -std=c++17 -MMD -MP -c $< -o $@

build/runtime/main.o: src/main.cpp | build/runtime
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -std=c++17 -MMD -MP -c $< -o $@

build/runtime/terminal.o: src/terminal/terminal.cpp | build/runtime
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -std=c++17 -MMD -MP -c $< -o $@

build/runtime:
	mkdir -p $@

repl: build/qwen_repl

build/qwen_repl: $(REPL_OBJS) $(RUNTIME_LIB) $(MODEL_LIB) $(TARGET) $(RESOURCE_LIB) $(TOKENIZER_LIB)
	$(NVCC) -arch=$(ARCH) $(NVCCFLAGS) $^ $(TOKENIZER_NVCC_LDLIBS) -o $@

-include $(RUNTIME_OBJS:.o=.d) $(REPL_OBJS:.o=.d)

.PHONY: repl

# 使用真实 tokenizer，替代 GPU 前向与资源函数检查控制流和异常路径。
VOCAB_PATH ?= ../llama-deploy/models/qwen2.5-1.5b-instruct-q8_0.gguf

test-runtime: build/runtime/runtime_test
	./build/runtime/runtime_test $(VOCAB_PATH)

build/runtime/runtime_test: tests/runtime_test.cpp $(RUNTIME_LIB) $(TOKENIZER_LIB)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -std=c++17 -UNDEBUG $^ $(TOKENIZER_LDLIBS) -o $@

.PHONY: test-runtime

build/runtime/session_gpu_test: tests/session_gpu_test.cu $(RUNTIME_LIB) $(MODEL_LIB) $(TARGET) $(RESOURCE_LIB) $(TOKENIZER_LIB)
	$(NVCC) -arch=$(ARCH) $(CPPFLAGS) $(NVCCFLAGS) -std=c++17 -UNDEBUG $^ $(TOKENIZER_NVCC_LDLIBS) -o $@
