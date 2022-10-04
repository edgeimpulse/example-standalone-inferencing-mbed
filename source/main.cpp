#include "mbed.h"

#include "benchmark.h"

#if BENCHMARK == BENCHMARK_BASELINE
    #define EI_CLASSIFIER_INFERENCING_ENGINE         EI_CLASSIFIER_NONE
    #define EI_CLASSIFIER_TFLITE_INPUT_DATATYPE      EI_CLASSIFIER_DATATYPE_INT8
    #define EI_CLASSIFIER_TFLITE_INPUT_QUANTIZED     1
    #define EI_CLASSIFIER_COMPILED                   0
#elif BENCHMARK == BENCHMARK_TFLITE_INT8
    #define EI_CLASSIFIER_INFERENCING_ENGINE         EI_CLASSIFIER_TFLITE
    #define EI_CLASSIFIER_TFLITE_INPUT_DATATYPE      EI_CLASSIFIER_DATATYPE_INT8
    #define EI_CLASSIFIER_TFLITE_INPUT_QUANTIZED     1
    #define EI_CLASSIFIER_COMPILED                   0
#elif BENCHMARK == BENCHMARK_TFLITE_FLOAT32
    #define EI_CLASSIFIER_INFERENCING_ENGINE         EI_CLASSIFIER_TFLITE
    #define EI_CLASSIFIER_TFLITE_INPUT_DATATYPE      EI_CLASSIFIER_DATATYPE_FLOAT32
    #define EI_CLASSIFIER_TFLITE_INPUT_QUANTIZED     0
    #define EI_CLASSIFIER_COMPILED                   0
#elif BENCHMARK == BENCHMARK_EON_INT8
    #define EI_CLASSIFIER_INFERENCING_ENGINE         EI_CLASSIFIER_TFLITE
    #define EI_CLASSIFIER_TFLITE_INPUT_DATATYPE      EI_CLASSIFIER_DATATYPE_INT8
    #define EI_CLASSIFIER_TFLITE_INPUT_QUANTIZED     1
    #define EI_CLASSIFIER_COMPILED                   1
#elif BENCHMARK == BENCHMARK_EON_FLOAT32
    #define EI_CLASSIFIER_INFERENCING_ENGINE         EI_CLASSIFIER_TFLITE
    #define EI_CLASSIFIER_TFLITE_INPUT_DATATYPE      EI_CLASSIFIER_DATATYPE_FLOAT32
    #define EI_CLASSIFIER_TFLITE_INPUT_QUANTIZED     0
    #define EI_CLASSIFIER_COMPILED                   1
#else
    #error "Unknown benchmark"
#endif

#include "edge-impulse-sdk/classifier/ei_run_classifier.h"
#include "edge-impulse-sdk/dsp/numpy.hpp"
#include <vector>

static const int MAX_NUMBER_STRING_SIZE = 32;
static char s[MAX_NUMBER_STRING_SIZE];

static const float dsp_features[1] = { 0 };

void print_memory_info() {
    // Grab the heap statistics
    mbed_stats_heap_t heap_stats;
    mbed_stats_heap_get(&heap_stats);
    printf("Heap size: %lu / %lu bytes (max: %lu)\r\n", heap_stats.current_size, heap_stats.reserved_size, heap_stats.max_size);
}

void ei_printf_float(float f) {
    float n = f;

    static double PRECISION = 0.00001;

    if (n == 0.0) {
        strcpy(s, "0");
    }
    else {
        int digit, m;
        char *c = s;
        int neg = (n < 0);
        if (neg) {
            n = -n;
        }
        // calculate magnitude
        m = log10(n);
        if (neg) {
            *(c++) = '-';
        }
        if (m < 1.0) {
            m = 0;
        }
        // convert the number
        while (n > PRECISION || m >= 0) {
            double weight = pow(10.0, m);
            if (weight > 0 && !isinf(weight)) {
                digit = floor(n / weight);
                n -= (digit * weight);
                *(c++) = '0' + digit;
            }
            if (m == 0 && n > 0) {
                *(c++) = '.';
            }
            m--;
        }
        *(c) = '\0';
    }

    printf("%s", s);
}

int main() {
    printf("Edge Impulse standalone inferencing (Mbed)\n");

    printf("Configuration:\n");
    printf("DSP feature size: %d\n", EI_CLASSIFIER_NN_INPUT_FRAME_SIZE);
    printf("Compiled: %d\n", EI_CLASSIFIER_COMPILED);

    // some things that we want present in the firmware
    printf("Test float: ");
    ei_printf_float(3.0f);
    printf("\n");

    // printf("read_ms returned %lu\n", ei_read_timer_ms());
    void *m = malloc(1);
    printf("address of m %p\n", m);
    free(m);

    ThisThread::sleep_for(1);

    printf("Initial memory usage:\n");
    print_memory_info();
    printf("\n\n");

#if BENCHMARK == BENCHMARK_BASELINE
    return 1;
#endif

    ei::matrix_t dsp_matrix(sizeof(dsp_features) / sizeof(dsp_features[0]), 1, (float*)dsp_features);

    while (1) {
        printf("Running inference\n");
        ei_impulse_result_t result = { 0 };

        static_assert(ei_dsp_blocks_size == 0, "Expected 'ei_dsp_blocks_size' to be 0, otherwise you'll get incorrect benchmark numbers");

        const ei_impulse_t impulse = ei_construct_impulse();
        EI_IMPULSE_ERROR r = run_inference(&impulse, &dsp_matrix, &result);
        if (r != EI_IMPULSE_OK) {
            printf("Failed to run impulse (%d)\n", r);
            ThisThread::sleep_for(2000);
            continue;
        }

        printf("Predictions (DSP: %d ms., Classification: %d ms., Anomaly: %d ms.): \n",
            result.timing.dsp, result.timing.classification, result.timing.anomaly);

        // print the predictions
        printf("[");
        for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
            ei_printf_float(result.classification[ix].value);
            if (ix != EI_CLASSIFIER_LABEL_COUNT - 1) {
                printf(", ");
            }
        }
        printf("]\n");

        print_memory_info();
        printf("\n");

        ThisThread::sleep_for(2000);
    }
}
