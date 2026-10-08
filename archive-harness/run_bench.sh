#!/bin/zsh
cd "$1"
while true; do
  l=$(sysctl -n vm.loadavg | awk '{print $2}' | tr ',' '.')
  if (( l < ${GATE:-6.0} )); then break; fi
  sleep 20
done
echo "start load $(sysctl -n vm.loadavg)" >&2
./target/release/time ${2:-10} $3 > "$4" 2> "$4.err"
echo "end load $(sysctl -n vm.loadavg)" >> "$4.err"
