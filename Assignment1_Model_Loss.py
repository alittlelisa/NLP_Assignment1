from torch.nn import L1Loss
from ignite.contrib.metrics.regression.r2_score import R2Score
from sklearn.metrics import mean_squared_error


def model_loss(model, dataset, train=False, optimizer=None):
    performance = L1Loss()
    score_metric = R2Score()

    avg_loss = 0
    avg_score = 0
    avg_mse = 0
    count = 0

    for input, output in iter(dataset):
        predictions = model.feed(input)

        loss = performance(predictions, output)

        score_metric.update([predictions, output])
        score = score_metric.compute()

        mse = mean_squared_error(output.cpu(), predictions.cpu().detach().numpy())

        if (train):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss += loss.item()
        avg_score += score
        count += 1
        avg_mse += mse

    return avg_loss / count, avg_score / count, avg_mse / count