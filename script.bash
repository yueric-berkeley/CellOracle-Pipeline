#!/bin/bash
CONFIG_FILE="config.yaml"

echo "[RunCellOracle] Running pipeline with $CONFIG_FILE"
python RunCellOracle.py --config $CONFIG_FILE
echo "[RunCellOracle] Finished."