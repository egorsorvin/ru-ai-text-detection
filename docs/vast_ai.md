# Running the full pipeline on a rented GPU (vast.ai)

Target: one GPU with 80 GB (H100 or A100 80 GB), on-demand, verified datacenter host,
reliability > 99.5 %, disk >= 150 GB. Expected wall time 7-9 h, cost ~20-25 $ on H100.

## 1. Rent

1. vast.ai -> Templates -> pick the official **PyTorch (CUDA)** template (Jupyter + SSH).
2. Filters: 1x GPU, GPU RAM >= 80 GB, disk >= 150 GB, on-demand, verified.
3. Rent. Wait until status is *running*, then copy the SSH command from the instance card
   (looks like `ssh -p 12345 root@ssh5.vast.ai -L 8080:localhost:8080`).

## 2. Connect from Windows PowerShell

```powershell
ssh -p <PORT> root@<HOST>
```

On the instance:

```bash
git clone https://github.com/<user>/ru-ai-text-detection.git
cd ru-ai-text-detection
mkdir -p outputs
nohup bash run_all.sh > outputs/run_all.log 2>&1 &
tail -f outputs/run_all.log        # Ctrl+C stops tail only, the job keeps running
```

Per-stage logs: `outputs/logs/*.log`. Check GPU: `nvidia-smi`.
Stage order: setup -> data -> generate -> features -> zero-shot eval -> mixed training -> summary.
Every stage skips work that is already on disk, so after any crash:

```bash
git pull && nohup bash run_all.sh >> outputs/run_all.log 2>&1 &
```

Environment knobs (prefix the command): `CAP=16000 GEN_N=1000 ENCODER=ai-forever/ruRoberta-large`.
To skip a generator or a pair, override `GENERATORS="..."` / `PAIRS="..."` (see the top of run_all.sh).

## 3. Bring results home (do this after every finished stage, not only at the end)

From PowerShell on the PC:

```powershell
scp -P <PORT> -r root@<HOST>:ru-ai-text-detection/outputs/results E:\ru-ai-text-detection\outputs\
scp -P <PORT> -r root@<HOST>:ru-ai-text-detection/outputs/scores  E:\ru-ai-text-detection\outputs\
scp -P <PORT> -r root@<HOST>:ru-ai-text-detection/outputs/logs    E:\ru-ai-text-detection\outputs\
scp -P <PORT> -r root@<HOST>:ru-ai-text-detection/data/gen        E:\ru-ai-text-detection\data\
```

Checkpoints (`outputs/checkpoints/*/best.pt`, 1.4 GB each) are optional; copy only if needed.

## 4. Stop paying

Instance card -> **Destroy** (not *Stop*: a stopped instance still bills disk).
