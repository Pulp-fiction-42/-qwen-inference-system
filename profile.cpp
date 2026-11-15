#include <nvtx3/nvToolsExt.h>
#include <thread>
#include <chrono>

int main() {
    nvtxRangePushA("Prefill");

    std::this_thread::sleep_for(
        std::chrono::milliseconds(100)
    );

    nvtxRangePop();
    return 0;
}
