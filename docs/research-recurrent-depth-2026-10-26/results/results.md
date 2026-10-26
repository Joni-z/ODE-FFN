# Recurrent-depth screen

All42 registered cells are terminal. Metrics below are independently recomputed from saved arrays. No new test targets were used.

| Dataset | Arm | Field MSE | Jacobian error | SW2 | Params | MACs |
|---|---|---:|---:|---:|---:|---:|
| gaussian_aniso_2d | ode_k2_native_final | 0.00444879 | 0.0180452 | 0.0797233 | 82056 | 105152 |
| gaussian_aniso_2d | swiglu192_native_final | 0.00420834 | 0.0267986 | 0.0772036 | 106818 | 104768 |
| gaussian_aniso_2d | ode_k2_native_deep | 0.00506417 | 0.0237783 | 0.0744438 | 82056 | 105152 |
| gaussian_aniso_2d | swiglu192_native_deep | 0.00398815 | 0.0168285 | 0.0785866 | 106818 | 104768 |
| gaussian_aniso_2d | ode_k2_cloned_final | 0.00479017 | 0.0186412 | 0.0778192 | 82056 | 105152 |
| gaussian_aniso_2d | swiglu192_cloned_final | 0.00449701 | 0.0278301 | 0.0749964 | 106818 | 104768 |
| gaussian_aniso_2d | ode_k2_cloned_deep | 0.00521246 | 0.0198053 | 0.0732438 | 82056 | 105152 |
| gaussian_aniso_2d | swiglu192_cloned_deep | 0.00397856 | 0.0181015 | 0.0807823 | 106818 | 104768 |
| gaussian_aniso_2d | ode_k2_shared_final | 0.00552919 | 0.0192655 | 0.0743423 | 31876 | 105152 |
| gaussian_aniso_2d | swiglu192_shared_final | 0.00462444 | 0.0252552 | 0.0769076 | 40130 | 104768 |
| gaussian_aniso_2d | ode_k2_shared_deep | 0.00609727 | 0.0175344 | 0.0763676 | 31876 | 105152 |
| gaussian_aniso_2d | swiglu192_shared_deep | 0.00442931 | 0.0194985 | 0.0800895 | 40130 | 104768 |
| gaussian_aniso_2d | swiglu128_native_final | 0.00392725 | 0.0259185 | 0.0779447 | 81792 | 80000 |
| gaussian_aniso_2d | swiglu128_native_deep | 0.00416146 | 0.0177315 | 0.0813958 | 81792 | 80000 |
| gmm_ring8_2d | ode_k2_native_final | 0.0325026 | 1.62537 | 0.27044 | 82056 | 105152 |
| gmm_ring8_2d | swiglu192_native_final | 0.0154476 | 0.540739 | 0.233545 | 106818 | 104768 |
| gmm_ring8_2d | ode_k2_native_deep | 0.0407415 | 1.92539 | 0.268372 | 82056 | 105152 |
| gmm_ring8_2d | swiglu192_native_deep | 0.019728 | 0.635866 | 0.241876 | 106818 | 104768 |
| gmm_ring8_2d | ode_k2_cloned_final | 0.029726 | 1.23626 | 0.271526 | 82056 | 105152 |
| gmm_ring8_2d | swiglu192_cloned_final | 0.0145355 | 0.684701 | 0.229192 | 106818 | 104768 |
| gmm_ring8_2d | ode_k2_cloned_deep | 0.0422596 | 1.84384 | 0.285044 | 82056 | 105152 |
| gmm_ring8_2d | swiglu192_cloned_deep | 0.0163687 | 0.617691 | 0.242661 | 106818 | 104768 |
| gmm_ring8_2d | ode_k2_shared_final | 0.0698944 | 2.51271 | 0.319896 | 31876 | 105152 |
| gmm_ring8_2d | swiglu192_shared_final | 0.015296 | 0.707992 | 0.230677 | 40130 | 104768 |
| gmm_ring8_2d | ode_k2_shared_deep | 0.0338066 | 1.34306 | 0.28625 | 31876 | 105152 |
| gmm_ring8_2d | swiglu192_shared_deep | 0.0174289 | 0.691348 | 0.241119 | 40130 | 104768 |
| gmm_ring8_2d | swiglu128_native_final | 0.0152963 | 0.736845 | 0.22636 | 81792 | 80000 |
| gmm_ring8_2d | swiglu128_native_deep | 0.0185973 | 0.597938 | 0.25017 | 81792 | 80000 |
| rotated_product_bimodal_16d | ode_k2_native_final | 0.325527 | 59.3056 | 0.124395 | 83862 | 106944 |
| rotated_product_bimodal_16d | swiglu192_native_final | 0.0704226 | 19.8841 | 0.0927089 | 108624 | 106560 |
| rotated_product_bimodal_16d | ode_k2_native_deep | 0.348996 | 62.3484 | 0.132712 | 83862 | 106944 |
| rotated_product_bimodal_16d | swiglu192_native_deep | 0.0487047 | 14.22 | 0.0930208 | 108624 | 106560 |
| rotated_product_bimodal_16d | ode_k2_cloned_final | 0.360862 | 63.6788 | 0.10494 | 83862 | 106944 |
| rotated_product_bimodal_16d | swiglu192_cloned_final | 0.0639749 | 18.1284 | 0.0994981 | 108624 | 106560 |
| rotated_product_bimodal_16d | ode_k2_cloned_deep | 0.383931 | 67.9769 | 0.122103 | 83862 | 106944 |
| rotated_product_bimodal_16d | swiglu192_cloned_deep | 0.0486151 | 14.0089 | 0.101181 | 108624 | 106560 |
| rotated_product_bimodal_16d | ode_k2_shared_final | 0.397258 | 67.8532 | 0.12904 | 33682 | 106944 |
| rotated_product_bimodal_16d | swiglu192_shared_final | 0.0895918 | 21.9228 | 0.10033 | 41936 | 106560 |
| rotated_product_bimodal_16d | ode_k2_shared_deep | 0.401976 | 67.6832 | 0.138319 | 33682 | 106944 |
| rotated_product_bimodal_16d | swiglu192_shared_deep | 0.0614098 | 15.1888 | 0.0970484 | 41936 | 106560 |
| rotated_product_bimodal_16d | swiglu128_native_final | 0.0923258 | 24.8899 | 0.0957215 | 83598 | 81792 |
| rotated_product_bimodal_16d | swiglu128_native_deep | 0.0409026 | 11.7333 | 0.10063 | 83598 | 81792 |

Training-depth nominees: none
Extended-generation eligibility: none

Sharing and intermediate supervision are distinct factors. Generic recurrence gains are not original-ODELayer gains. Recorded training seconds include concurrent evaluation and are not isolated latency measurements.
