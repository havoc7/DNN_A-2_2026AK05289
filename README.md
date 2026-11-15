# DNN Assignment 2 – RNN, LSTM, GRU and Transformer

**Name:** Sarthak Rout  **Student ID:** 2026AK05289

## Environment
- Python, PyTorch, NumPy, pandas, Matplotlib, scikit-learn (exact versions in `requirements.txt`).
- Run on Google Colab on Runtime set to T4.

## My variant - couldn't find the variant table or the individualization thing mentioned so just continued with odd last digit of BITS ID
- Hidden size (Problem 1): 64
- Seeds: <0, 1, 2>
- Ablation 5(c) (last digit of ID = 9): LSTM forget-gate bias 0 vs 1
- Transformer sizes (Problem 2): d = 128, 4 heads, 4 layers, d_ff = 512, context 128

## How to run
1. Unzip, keep the folder structure as is. Datasets are NOT included; the notebooks download them into `../data` on first run (needs internet).
2. Open `problem1/problem1.ipynb` with the working directory set to `problem1/` and run *Restart and Run All*.
3. Then open `problem2/problem2.ipynb` (working directory `problem2/`) and run *Restart and Run All*. Run Problem 1 first: Problem 2 reads `../results.csv` for the HAR comparison.
4. Outputs: `results.csv` (root, appended by both notebooks), `problem1/figures/`, `problem2/figures/`, `problem2/generated_samples.txt`.
- If a download is blocked, use the instructor's mirror and place the files under `data/` (`data/har/UCI HAR Dataset/…`, `data/tinyshakespeare.txt`).

## Total run time on a T4
- Problem 1: 14 minutes
- Problem 2: 18 minutes

## Assumptions
- Individualisation values - couldn't find the variant table or the individualization thing mentioned so just continued with odd last digit of BITS ID.
- Adding problem: all models trained with gradient clipping (norm 1.0); the 5(c) ablation compares against LSMT forget bias 0 vs 1.
- HAR: 30 epochs, adding problem: 40 epochs, Adam lr 1e-3 for all recurrent models; same budget for RNN, LSTM, GRU.
- GRU/LSTM/RNN cells use one bias vector per gate; for the library comparison my weights are copied into `nn.*` with `bias_hh = 0` (needed for the GRU's reset-gate bias).
- Task A: 2 encoder layers, 4 heads, patches of 8 steps, AdamW lr 5e-4. Task B: one seed, AdamW lr 1e-3 with warmup and cosine decay, top-k 40, sinusoidal positions.
- Bonus (key-value cache / LSTM character model): Not attempted.

## Known issues
- None
