#include "mbed.h"
#include "ei_run_classifier.h"
#include "numpy.hpp"

#define MODE_F32        1
#define MODE_I16        2
#define INFERENCE_MODE  MODE_I16


static const float features[] = {
    6.9000, -0.3500, -6.4600, 6.9000, -0.3500, -6.4600, 6.8700, 0.2900, -5.3800, 8.6100, -0.8700, -5.8700, 16.8200, -2.3000, -7.1400, 19.9700, -5.1700, -7.9100, 19.9700, -5.5200, -6.1500, 19.9700, -4.5100, -3.4400, 19.9700, -4.5100, -3.4400, 19.9700, -5.0200, -2.7100, 19.9700, -6.3400, -3.5500, 19.9700, -7.9300, -4.8500, 19.9700, -6.9700, -4.0600, 15.5300, -4.9000, -3.8300, 12.8200, -4.9800, -4.5400, 10.9200, -4.0000, -4.1400, 8.5400, -2.3300, -3.7800, 8.5400, -2.3300, -2.4800, 4.4700, -2.1900, -2.4800, 3.5300, -0.7200, -2.3800, 1.9200, -1.0700, -2.1900, 0.3300, -1.0300, -1.5800, -0.6400, -0.6100, -0.0800, -3.5900, -0.2100, 0.2900, -3.5900, -0.2100, 0.2900, -6.4800, 1.1600, 2.7500, -6.5300, -0.6100, 2.2000, -8.9500, 0.0200, 2.7200, -12.5400, -0.1800, 4.4100, -15.2200, 2.2300, 6.6400, -19.3700, 3.3200, 9.5600, -19.3700, 3.3200, 9.5600, -19.9800, 4.0100, 11.9700, -19.9800, 3.4800, 11.9000, -19.9800, 2.9800, 11.8500, -19.9800, 2.3300, 12.1500, -19.4000, 0.6000, 11.4500, -17.7800, -0.5800, 9.8800, -17.7800, -0.5800, 9.8800, -16.0100, -1.4400, 8.0000, -13.4500, -2.1700, 6.1000, -12.2600, -2.6800, 4.9800, -10.6400, -3.3200, 3.1800, -8.5700, -3.7200, 1.7900, -7.0000, -2.9500, 0.7800, -7.0000, -2.9500, 0.7800, -5.8400, -1.0900, -0.1400, -3.7200, -0.2000, -1.8200, -2.3900, 0.6000, -2.7300, -0.6700, 0.9800, -3.9100, 2.1500, 1.3100, -5.7300, 5.6000, 1.0500, -7.2100, 5.6000, 1.0500, -7.2100, 10.3900, 0.4300, -7.6200, 16.0000, -0.4500, -7.1300, 19.7800, -3.3700, -7.2600, 19.9700, -5.6500, -6.7300, 19.9700, -6.7400, -5.7400, 19.9700, -6.0700, -4.1100, 19.9700, -6.0700, -4.1100, 19.9700, -6.3100, -3.5800, 19.9700, -7.5200, -4.7200, 19.9700, -7.4600, -5.0500, 19.6500, -6.4000, -4.8700, 16.2400, -4.6200, -3.9500, 13.8200, -4.4700, -4.1300, 13.8200, -4.4700, -4.1300, 10.8500, -4.5300, -4.3600, 8.4300, -4.2700, -4.1800, 6.1100, -3.3700, -3.1500, 4.6100, -2.4500, -2.4100, 2.9700, -2.9900, -2.9400, 1.1300, -2.5800, -2.7200, 1.1300, -2.5800, -2.7200, 0.6000, -1.3000, -1.4400, -1.3600, 0.3400, -0.2300, -4.4600, 1.0000, 0.9200, -5.9500, 0.3200, 1.4200, -5.7000, 0.6300, 1.9600, -5.7000, 0.6300, 1.9600, -7.9100, 1.2900, 2.6300, -11.7400, 2.1000, 4.2300, -16.0500, 2.6200, 6.9200, -18.9500, 1.4000, 8.6500, -19.9800, 0.1400, 10.1200, -19.9800, -0.4500, 10.6000, -19.9800, -0.4500, 10.6000, -19.9800, -1.0500, 10.8300, -19.9800, -2.7100, 10.3000, -19.9800, -5.3900, 8.7400, -19.9100, -7.2100, 6.8000, -18.1400, -7.0400, 5.9700, -17.4200, -5.5800, 5.8500, -17.4200, -5.5800, 5.8500, -15.9800, -3.5500, 5.7300, -12.6700, -2.3300, 4.2400, -10.3300, -1.1600, 3.2400, -8.8200, -0.7800, 2.5300, -7.3700, -0.1200, 1.9900, -7.9300, 1.2100, 2.0600, -7.9300, 1.2100, 2.0600, -6.6800, 1.7100, 0.9000, -4.0500, 1.2500, -1.1500, -0.6300, 1.3500, -2.6000, 0.6900, 1.3000, -3.4900, 1.2400, 2.1600, -3.8900, 1.8000, 3.9300, -3.3000, 1.8000, 3.9300, -3.3000, 7.5700, 2.8500, -4.9900, 15.7800, -0.0800, -7.2700, 19.9700, -4.1800, -8.9600, 19.9700, -5.3500, -7.5200, 19.9700, -4.6100, -5.6800, 19.9700, -6.4500, -5.7300, 19.9700, -6.4500, -5.7300, 19.9700, -8.3300, -6.7300, 19.9700, -8.6000, -6.5700, 19.9700, -7.5200, -6.2900, 19.8900, -6.5600, -5.7300, 16.9000, -5.5900, -5.1300, 13.9300, -5.3300, -4.9100, 13.9300, -5.3300, -4.9100, 11.4000, -5.1100, -5.0700, 10.0100, -4.3100, -5.0500, 7.4600, -2.8300, -4.4400
};

#if INFERENCE_MODE == MODE_I16
int raw_feature_get_data(size_t offset, size_t length, int16_t *out_ptr) {
    for (size_t ix = 0; ix < length; ix++) {
        float v = features[offset + ix];
        float scaled_float = v / 19.62;
        arm_float_to_q15(&scaled_float, (q15_t *)&out_ptr[ix], 1);
    }
    return 0;
}
#else
int raw_feature_get_data(size_t offset, size_t length, float *out_ptr) {
    memcpy(out_ptr, features + offset, length * sizeof(float));
    return 0;
}
#endif

int main() {
    printf("Edge Impulse standalone inferencing (Mbed)\n");

    if (sizeof(features) / sizeof(float) != EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE) {
        printf("The size of your 'features' array is not correct. Expected %d items, but had %u\n",
            EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, sizeof(features) / sizeof(float));
        return 1;
    }

    ei_impulse_result_t result = { 0 };

    while (1) {
        // the features are stored into flash, and we don't want to load everything into RAM
#if INFERENCE_MODE == MODE_I16
        signal_i16_t features_signal;
        features_signal.total_length = sizeof(features) / sizeof(features[0]);
        features_signal.get_data_i16 = &raw_feature_get_data;

        // invoke the impulse
        EI_IMPULSE_ERROR res = run_classifier_i16(&features_signal, &result, true);
#else
        // the features are stored into flash, and we don't want to load everything into RAM
        signal_t features_signal;
        features_signal.total_length = sizeof(features) / sizeof(features[0]);
        features_signal.get_data = &raw_feature_get_data;

        // invoke the impulse
        EI_IMPULSE_ERROR res = run_classifier(&features_signal, &result, true);
#endif

        printf("run_classifier returned: %d\n", res);

        if (res != 0) return 1;

        printf("Predictions (DSP: %d ms., Classification: %d ms., Anomaly: %d ms.): \n",
            result.timing.dsp, result.timing.classification, result.timing.anomaly);

        // print the predictions
        printf("[");
        for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
            printf("%.5f", result.classification[ix].value);
#if EI_CLASSIFIER_HAS_ANOMALY == 1
            printf(", ");
#else
            if (ix != EI_CLASSIFIER_LABEL_COUNT - 1) {
                printf(", ");
            }
#endif
        }
#if EI_CLASSIFIER_HAS_ANOMALY == 1
        printf("%.3f", result.anomaly);
#endif
        printf("]\n");

        ThisThread::sleep_for(2000);
    }
}
