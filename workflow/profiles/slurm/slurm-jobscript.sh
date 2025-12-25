#!/bin/bash
#SBATCH --job-name={rule}
#SBATCH --cpus-per-task={threads}
#SBATCH --time={resources.time}
#SBATCH --output=slurm_logs/{rule}-%j.out
#SBATCH --error=slurm_logs/{rule}-%j.err

# --- memory handling ---
{resources.mem_mb and f"#SBATCH --mem={resources.mem_mb}" or ""}
{resources.mem_per_cpu_mb and f"#SBATCH --mem-per-cpu={resources.mem_per_cpu_mb}" or ""}

# --- optional cluster params ---
{resources.partition and f"#SBATCH --partition={resources.partition}" or ""}
{resources.qos and f"#SBATCH --qos={resources.qos}" or ""}
{resources.account and f"#SBATCH --account={resources.account}" or ""}

set -euo pipefail

echo "## ========================================="
echo "## Running rule: {rule}"
echo "## Job ID: $SLURM_JOB_ID"
echo "## Host: $(hostname)"
echo "## Started at: $(date)"
echo "## ========================================="

LOGFILE=logs/{rule}-%j.log
mkdir -p logs
exec > >(tee -a "$LOGFILE") 2>&1

trap 'echo "## ERROR: Rule {rule} failed on host $(hostname) at $(date)" >&2' ERR

/usr/bin/time -v bash -c "{exec_job}"

echo "## ========================================="
echo "## Finished rule: {rule}"
echo "## Completed at: $(date)"
echo "## ========================================="

if command -v sacct &>/dev/null; then
    echo "## Resource usage from Slurm (sacct):"
    sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed,MaxRSS,MaxVMSize,ReqMem,AllocCPUS
fi
