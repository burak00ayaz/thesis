#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/../.."

# python3 -m experiments.indel.pisco_nq
# python3 -m experiments.indel.pisco_bioasq
# python3 -m experiments.indel.mistral_nq
# python3 -m experiments.indel.mistral_bioasq
# python3 -m experiments.indel.xrag_nq
# python3 -m experiments.indel.xrag_bioasq
python3 -m experiments.indel.pisco_squad
python3 -m experiments.indel.mistral_squad
python3 -m experiments.indel.xrag_squad