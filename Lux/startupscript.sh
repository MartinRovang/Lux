

uvicorn  lux:app --host 0.0.0.0 --port 8080 --reload &
P1=$!
python3 lux/src/tools/backgroundprocess2.py &
P2=$!
wait $P1 $P2