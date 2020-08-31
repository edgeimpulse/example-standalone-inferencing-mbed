#include "mbed.h"
#include "ei_run_classifier.h"
#include "numpy.hpp"

static const int MAX_NUMBER_STRING_SIZE = 32;
static char s[MAX_NUMBER_STRING_SIZE];

static const float dsp_features[] = {
    0.9922, 0.9921, 1.1408, 2.4802, 0.2499, 3.4722, 0.1487, 0.0503, 0.1358, 0.0015, 0.0018, 1.6157, 0.9921, 1.3641, 3.4722, 0.4255, 5.4563, 0.1007, 0.0193, 0.1938, 0.1085, 0.0151, 5.9337, 0.9921, 7.0465, 2.4802, 1.5141, 3.9683, 0.3771, 1.1192, 5.1703, 0.1974, 0.0571
};

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

    if (sizeof(dsp_features) / sizeof(float) != EI_CLASSIFIER_NN_INPUT_FRAME_SIZE) {
        printf("The size of your 'dsp_features' array is not correct. Expected %d items, but had %u\n",
            EI_CLASSIFIER_NN_INPUT_FRAME_SIZE, sizeof(dsp_features) / sizeof(float));
        return 1;
    }

    ei::matrix_t dsp_matrix(sizeof(dsp_features) / sizeof(dsp_features[0]), 1, (float*)dsp_features);

    while (1) {
        ei_impulse_result_t result = { 0 };

        EI_IMPULSE_ERROR r = run_inference(&dsp_matrix, &result);
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

        ThisThread::sleep_for(2000);
    }
}
