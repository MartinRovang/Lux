

uvicorn  lux:app --host 0.0.0.0 --port 8080 --reload &
P1=$!
python3 lux/src/tools/backgroundprocess1.py &
P2=$!
python3 lux/src/tools/backgroundprocess2.py &
P3=$!
wait $P2 $P1 $P3