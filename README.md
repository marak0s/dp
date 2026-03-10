# Neural Viterbi vs Neural BCJR (AWGN)

Воспроизводимый пакет для сравнения baseline-декодеров (Viterbi, BCJR) и прототипов learned-декодеров (LLR-calibrated Neural Viterbi, LLR-calibrated Neural BCJR) для сверточных кодов в AWGN.

## Установка

```bash
pip install -e .
pip install -e ".[dev,notebooks]"
# optional
pip install -e ".[dev,notebooks,tf]"
```

## Быстрый старт (CLI)

```bash
python -m comm_ai.experiments.run_experiment --config src/comm_ai/config/experiments/awgn_small.yaml
```

## Как обучить neural-модели

```bash
python -m comm_ai.training.train_neural_viterbi --config src/comm_ai/config/experiments/awgn_small.yaml
python -m comm_ai.training.train_neural_bcjr --config src/comm_ai/config/experiments/awgn_small.yaml
```

Также `run_experiment` может автоматически обучать модели, если `training.enabled: true`.

## Как использовать сохранённые сигналы

1. Запустить эксперимент и получить `signals.npz`.
2. Включить `experiment.reuse_saved_signals: true` в YAML.
3. Повторно запустить `run_experiment` — данные будут переиспользованы.

## Как переиспользовать чекпоинты

- По умолчанию чекпоинты ищутся в `outputs/runs/<run_name>/checkpoints/`.
- Можно задать явные пути через `checkpoint_paths` в конфиге.
- Если чекпоинта нет и `training.enabled: false`, neural-декодер будет пропущен с предупреждением.

## Артефакты запуска

В `outputs/runs/<run_name>/` сохраняются:
- `signals.npz`
- `results.csv`
- `ber_plot.png`
- `fer_plot.png`
- `timing_plot.png`
- `summary.md`
- `config_used.yaml`
- `run_metadata.json`
- `checkpoints/best_neural_viterbi.pt`
- `checkpoints/best_neural_bcjr.pt`

## Запуск в Google Colab

```python
!git clone https://github.com/<your-org>/<your-repo>.git
%cd <your-repo>
!pip install -e ".[notebooks]"
!python -m comm_ai.experiments.run_experiment --config src/comm_ai/config/experiments/awgn_small.yaml
```

## Важная оговорка по learned-моделям

Текущая версия learned-декодеров — это **LLR calibration prototype**:
- neural-сеть калибрует LLR/ветвевые метрики;
- далее используется классический Viterbi/BCJR core.

Это не заявляется как полностью differentiable end-to-end trellis decoder core.
