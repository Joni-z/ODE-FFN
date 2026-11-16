# Statistics and comparison units

Training recipe: seed 0, 20,000 updates, batch 512, native AdamW 0.0003. Each
resolution uses the same trained weights and eight shared 4,096-source banks.
The pooled metric sorts all 32,768 generated/target samples together. Its value
is distinct from the average of eight separately computed bank SW2 values.

The primary descriptive family is palindrome minus alternating over two tasks
and five registered numerical settings. Bank identity is the paired sampling
unit. Numerical settings repeat those banks; they are not additional
independent observations. No checkpoint or setting is selected from this sweep.

Intervals enumerate all 6,435 multinomial bank-count vectors with multiplicities
summing to 8^8. Marginal 95% percentile and nominal ten-contrast Bonferroni
percentile intervals are retained in comparison.json. Bootstrap coverage is
approximate with eight empirical banks. The nonparametric paired sign test
uses nonzero signs and Holm correction over the ten comparisons. It addresses
a different null than the mean-difference bootstrap. No Holm-adjusted sign
p-value is below 0.05. The conclusions describe the observed fixed-checkpoint
ordering and its numerical persistence.

| Task | Inner/Heun | Mean paired difference | SD | Marginal 95% interval | Pal better banks | Holm sign p |
|---|---|---:|---:|---|---:|---:|
| Ring8 | 4/128 | -0.012 | 0.009 | [-0.017,-0.006] | 7/8 | 0.492 |
| Ring8 | 4/256 | -0.012 | 0.009 | [-0.018,-0.006] | 7/8 | 0.492 |
| Ring8 | 4/512 | -0.012 | 0.009 | [-0.017,-0.006] | 7/8 | 0.492 |
| Ring8 | 8/128 | -0.012 | 0.010 | [-0.018,-0.005] | 7/8 | 0.492 |
| Ring8 | 16/128 | -0.012 | 0.010 | [-0.018,-0.005] | 7/8 | 0.492 |
| Product16 | 4/128 | 0.004 | 0.003 | [0.002,0.006] | 0/8 | 0.078 |
| Product16 | 4/256 | 0.004 | 0.003 | [0.002,0.006] | 1/8 | 0.492 |
| Product16 | 4/512 | 0.004 | 0.003 | [0.002,0.006] | 1/8 | 0.492 |
| Product16 | 8/128 | 0.008 | 0.003 | [0.006,0.009] | 0/8 | 0.078 |
| Product16 | 16/128 | 0.010 | 0.003 | [0.008,0.012] | 0/8 | 0.078 |

The two sweeps vary one axis at a time at frozen weights. They do not test
joint inner/outer limits or retraining at each inference budget. Ordinary
references retain their original optimizer labels. All full-precision effects,
intervals, source/array/checkpoint hashes and numerical verification are in
the adjacent JSON/CSV artifacts.

A separate validator enumerates sorted bootstrap-index tuples, uses integer
CDF thresholds and fsum, and recomputes all means, SDs, interval endpoints,
binomial probabilities and Holm adjustments directly from the bank CSV. Its
maximum difference is 3.47e-18; independent-statistics.json binds inputs/source.
