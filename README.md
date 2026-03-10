# Neural Viterbi vs Neural BCJR (AWGN)

Репозиторий для воспроизводимого сравнения baseline (Viterbi/BCJR) и обучаемых модификаций (Neural Viterbi/Neural BCJR) для сверточных кодов в AWGN.

## Быстрый старт

```bash
pip install -e .
python -m comm_ai.experiments.run_experiment --config src/comm_ai/config/experiments/awgn_small.yaml
```

## Что сохраняется

В `outputs/runs/<run_name>/`:
- `signals.npz`
- `results.csv`
- `ber_plot.png`, `fer_plot.png`, `timing_plot.png`
- `config_used.yaml`
- `summary.md`

## Запуск в Colab

```python
!git clone <repo_url>
%cd repo
!pip install -e .
!python -m comm_ai.experiments.run_experiment --config src/comm_ai/config/experiments/awgn_small.yaml
```

## Структура

Основной пакет расположен в `src/comm_ai` и включает:
- `channel/`, `codes/`, `decoders/`
- `datasets/` для сохранения/загрузки сигналов
- `training/` для обучения нейронных компонентов
- `experiments/` для CLI и оценки
