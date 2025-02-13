# EC25 Random-Probing Raccoon

This repository contains all the scripts necessary to **reproduce the results** presented in the paper:

**"New Techniques for Random Probing Security and Application to Raccoon Signature Scheme"**
by *Sonia Belaïd, Matthieu Rivain, and Mélissa Rossi*, published at *Eurocrypt 2025*.

## 📌 Overview

The provided Python script allows for:

- **Computing envelopes** of elementary and meta-gadgets used in the Raccoon signature scheme.
- **Evaluating the Random Probing Composability (RPC) security** of key generation and signature algorithms.
- **Measuring computational complexity** of individual gadgets and the overall protocol.
- **Reproducing numerical results and generating visual graphs** as presented in the paper.

## 📜 Features

### 🔹 Basic Gadget Envelopes

The script includes functions to compute envelopes of elementary gadgets:

- `cardinal_rpc_refresh_envelope(n, p, nb_iter)`: Computes the refresh gadget envelope with:

  - `n`: Number of shares.
  - `p`: Leaking wire probability.
  - `nb_iter`: Number of random values (*gamma*).

- `cardinal_rpc_add_envelope(n, p, pgref)`: Computes the addition gadget envelope using:

  - `pgref`: Envelope of the underlying refresh gadget.

- `cardinal_rpc_gcopy_envelope_pgref(n, pgref)`: Computes the copy gadget envelope.

- `cardinal_rpc_gcmult_envelope_pgref(n, p, pgref)`: Computes the multiplication gadget envelope.

### 🔹 Meta-Gadget Envelopes

- Envelopes for complex meta-gadgets such as **trees of copies** and **AddNoiseTo**.

### 🔹 Security Evaluation of Raccoon

- `rpc_keygen(n, t, tri, p, gamma1, gamma2, gamma3)`: Computes the **RPC advantage** for Raccoon key generation.
- `rpc_sign(n, t, tri, p, gamma1, gamma2, gamma3)`: Computes the **RPC advantage** for Raccoon signature generation.

### 🔹 Complexity Computation

Functions to compute the **computational cost** of:

- Individual gadgets depending on different **gamma values**.
- Key generation and signature in **Raccoon-128-16**.

## 🛠️ Dependencies

This script requires:

- Python 3.x
- `numpy`, `matplotlib`, `scipy`, `sympy`

To install dependencies, run:

```sh
pip install numpy matplotlib scipy sympy
```

## ▶️ Usage

Ensure the script includes a main entry point with function calls, such as:

```python
if __name__ == "__main__":
    # Example function calls
    print("Computing RPC envelopes...")
    rpc_keygen(n=16, t=8, tri=15, p=2**-16, gamma1=80, gamma2=20, gamma3=70)
    rpc_sign(n=16, t=8, tri=15, p=2**-16, gamma1=80, gamma2=20, gamma3=70)
```

Then, simply run the script:

```sh
python EC25-random-probing-Raccoon.py
```

To visualize graphs (if implemented in the script):

```sh
python EC25-random-probing-Raccoon.py --plot
```

## 📄 License

This project is distributed under the **MIT License**.

---

For any questions, please refer to the **paper** or contact the **authors**.
