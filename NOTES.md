Defender Training Run

```
python qlearning/orchestrator.py \
  --role defender \
  --games 50000 \
  --eval-interval 2000 \
  --eval-games 300 \
  --checkpoint-every 2000 \
  --opponent-policy random
```
Attacker Training Run

```
python qlearning/orchestrator.py \
  --role attacker \
  --games 50000 \
  --eval-interval 2000 \
  --eval-games 300 \
  --checkpoint-every 2000
  --learner-qtable-path qlearning/qtable_attacker.json \
  --learner-learning-log-path qlearning/learning_metrics_attacker.jsonl \
  --eval-learning-log-path qlearning/eval_metrics_attacker.jsonl
```

Frozen Attacker/Defender
```
python qlearning/orchestrator.py \
  --role defender \
  --games 10000 \
  --eval-interval 1000 \
  --checkpoint-every 2000 \
  --checkpoint-dir qlearning/checkpoints/def_crosstrain
  --learner-qtable-path qlearning/models/qtable_defender.json \
  --learner-learning-log-path qlearning/metrics/learning_metrics_crosstrain_defender.jsonl \ 
  --learner-username ct-defender
  --opponent-username ct-attacker
  --opponent-policy pool \
  --opponent-checkpoint-glob 'qlearning/checkpoints/attacker/attacker_*.json' \
  --opponent-pool-order random
```