#pragma once

#include "edge-impulse-sdk/dsp/spectral/spectral.hpp"
#include "edge-impulse-sdk/dsp/speechpy/speechpy.hpp"

int test() {
    ei::matrix_t matrix(30, 10);
    auto bla = ei::speechpy::processing::cmvnw(&matrix);

    float b[3];
    numpy::roll(b, 3, -1);

    ei::spectral::processing::scale(b, 3, 1.2);

    return bla;
}
