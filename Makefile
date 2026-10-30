NVCC := nvcc
ARCH := -arch=sm_120
NVCCFLAGS := -g -G

OBJDIR := make
TARGET := engine

OBJS := $(OBJDIR)/model.o $(OBJDIR)/forward.o $(OBJDIR)/layer.o $(OBJDIR)/run.o

TEST_KERNEL_SRCS := $(wildcard test_kernel/*.cu)
TEST_KERNEL_BINS := $(TEST_KERNEL_SRCS:.cu=)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(ARCH) $(OBJS) -o $@

$(OBJDIR)/model.o: model.cu model.h | $(OBJDIR)
	$(NVCC) $(ARCH) $(NVCCFLAGS) -c $< -o $@

$(OBJDIR)/forward.o: forward.cu forward.h layer.h model.h | $(OBJDIR)
	$(NVCC) $(ARCH) $(NVCCFLAGS) -c $< -o $@

$(OBJDIR)/layer.o: layer.cu layer.h forward.h model.h | $(OBJDIR)
	$(NVCC) $(ARCH) $(NVCCFLAGS) -c $< -o $@

$(OBJDIR)/run.o: run.cu forward.h model.h | $(OBJDIR)
	$(NVCC) $(ARCH) $(NVCCFLAGS) -c $< -o $@

# 针对 test_kernel 目录下测试可执行文件的编译规则
test_kernel/%: test_kernel/%.cu forward.h layer.h model.h $(OBJDIR)/forward.o $(OBJDIR)/layer.o $(OBJDIR)/model.o
	$(NVCC) $(ARCH) $(NVCCFLAGS) -I. $< $(OBJDIR)/forward.o $(OBJDIR)/layer.o $(OBJDIR)/model.o -o $@

test_kernel: $(TEST_KERNEL_BINS)

$(OBJDIR):
	mkdir -p $(OBJDIR)

clean:
	rm -f $(TARGET) $(OBJS) $(OBJDIR)/engine $(TEST_KERNEL_BINS)

.PHONY: all clean test_kernel
