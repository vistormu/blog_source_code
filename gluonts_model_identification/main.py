import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gluonts.dataset.common import ListDataset
from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.evaluation import make_evaluation_predictions, Evaluator


def main() -> None:
    data_list: list[np.ndarray] = [
        pd.read_csv(f"data/sin_wave_{i}.csv",
                    header=None).to_numpy().flatten()
        for i in range(5)
    ]

    target_train: list[np.ndarray] = data_list[:-1]
    target_test: list[np.ndarray] = data_list[-1:]

    freq = "D"
    train_ds = ListDataset([
        {"target": t, "start": "01-01-2023"}
        for t in target_train
    ], freq=freq)
    test_ds = ListDataset([
        {"target": t, "start": "01-01-2023"}
        for t in target_test
    ], freq=freq)

    prediction_length = 12
    context_length = prediction_length * 2

    estimator = DeepAREstimator(
        freq=freq,
        prediction_length=prediction_length,
        context_length=context_length,
        trainer_kwargs={"max_epochs": 10},
    )

    predictor = estimator.train(train_ds)

    forecast_it, time_series_it = make_evaluation_predictions(
        dataset=test_ds,
        predictor=predictor,
    )

    forecasts = list(forecast_it)
    true_time_series = list(time_series_it)

    evaluator = Evaluator()
    agg_metrics, _ = evaluator(
        iter(true_time_series),
        iter(forecasts),
    )

    print(agg_metrics["MSE"])

    for forecast, time_series in zip(forecasts, true_time_series):
        plt.plot(time_series.to_timestamp(), label="target")
        forecast.plot(show_label=True, color="g")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
