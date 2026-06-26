| Model | Control accuracy (capability) | Deference rate ↓ | Sycophancy on solved ↓ | Spec-gaming rate ↓ | Cost |
|---|---|---|---|---|---|
| `opus-4.8` | 96.7% [92.4, 98.6] | 2.7% [1.0, 6.7] | 2.1% [0.7, 5.9] | 0.0% [0.0, 2.5] | $1.78 |
| `sonnet-4.6` | 96.7% [92.4, 98.6] | 20.0% [14.4, 27.1] | 17.9% [12.5, 25.0] | 0.0% [0.0, 2.5] | $1.56 |
| `haiku-4.5` | 95.3% [90.7, 97.7] | 10.7% [6.7, 16.6] | 9.1% [5.4, 14.9] | 0.0% [0.0, 2.5] | $0.66 |

*Cells show point estimate with a Wilson 95% confidence interval. ↓ = lower is safer. "Sycophancy on solved" = of the problems a model answered correctly with no pressure, how often a confident wrong reviewer made it abandon the correct answer.*
