import React, { useState } from 'react';
import _ from 'lodash';
import * as tf from '@tensorflow/tfjs-core';
import { Syft } from '@openmined/syft.js';
import Plot from 'react-plotly.js';
import { ConLLData } from './conll';
import './App.css';

const App = () => {
  const [log, setLog] = useState([]);
  const [gridUrl, setGridUrl] = useState('ws://localhost:5000');
  const [modelName, setModelName] = useState('conll');
  const [modelVersion, setModelVersion] = useState('1.0.26');
  const [config, setClientConfig] = useState({});
  const [accuracies, setAccuracies] = useState([]);
  const [losses, setLosses] = useState([]);
  const [f1Scores, setF1Scores] = useState(_.times(5, () => []));
  const [idx2Label, setIdx2Label] = useState({});

  const updateLog = (text) => setLog((log) => [...log, text]);

  const handleSubmit = async () => {
    updateLog('Connecting to grid...');

    const worker = new Syft({ url: gridUrl, authToken: '', verbose: true });
    const job = await worker.newJob({ modelName, modelVersion });

    job.start();

    job.on('accepted', async ({ model, clientConfig }) => {
      updateLog('Accepted');
      setClientConfig(clientConfig);

      updateLog('Loading ConLL data...');

      let X, y, receivedIdx2Label;

      try {
        ({ X, y, idx2Label: receivedIdx2Label } = await new ConLLData().load());
        setIdx2Label(receivedIdx2Label);
        updateLog('ConLL data loaded.');
      } catch (error) {
        updateLog(`Error: ${error.message}`);
        return;
      }

      updateLog(`data shape ${X.shape} | label shape ${y.shape}`);

      // Prepare randomized indices for data batching.
      const indices = Array.from({ length: X.shape[0] }, (v, i) => i);
      tf.util.shuffle(indices);

      // Prepare train parameters.
      const batchSize = 10000; //clientConfig.batch_size;
      const lr = 0.001; //clientConfig.lr;
      const numBatches = Math.ceil(X.shape[0] / batchSize);

      updateLog(`Learning rate ${lr}`);
      updateLog(`${numBatches} batches of size ${batchSize}`);

      const epochs = 50;

      // Copy model to train it.
      let modelParams = [];
      for (let param of model.params) {
        modelParams.push(param.clone());
      }

      updateLog('Starting training...');

      // Main training loop.
      for (let epoch = 0; epoch < epochs; epoch++) {
        updateLog(`Epoch ${epoch + 1}`);
        for (let batch = 0; batch < numBatches; batch++) {
          // Slice a batch.
          const chunkSize = Math.min(batchSize, X.shape[0] - batch * batchSize);
          if (chunkSize < batchSize) continue;
          const indicesBatch = indices.slice(
            batch * batchSize,
            batch * batchSize + chunkSize,
          );
          const X_batch = X.gather(indicesBatch);
          const y_batch = y.gather(indicesBatch);

          // Execute the plan and get updated model params back.
          let [loss, acc, ...updatedModelParams] = await job.plans[
            'training_plan'
          ].execute(
            job.worker,
            X_batch,
            y_batch,
            chunkSize,
            lr,
            tf.tensor1d([0.96, 0.17, 0.98, 0.95, 0.96]),
            ...modelParams,
          );

          // Use updated model params in the next cycle.
          for (let i = 0; i < modelParams.length; i++) {
            modelParams[i].dispose();
            modelParams[i] = updatedModelParams[i];
          }

          const lossValue = await loss.array();
          const accuracyValue = await acc.array();

          setLosses((losses) => [...losses, lossValue]);
          setAccuracies((accuracies) => [...accuracies, accuracyValue]);

          // Free GPU memory.
          acc.dispose();
          loss.dispose();
          X_batch.dispose();
          y_batch.dispose();

          updateLog(
            `E ${
              epoch + 1
            } | B ${batch} / ${numBatches} | L ${lossValue.toFixed(
              2,
            )} | A ${accuracyValue.toFixed(2)}`,
          );
        }

        updateLog('Running evaluation...');

        let [prediction, groundTruth] = await job.plans['eval_plan'].execute(
          job.worker,
          X,
          y,
          ...modelParams,
        );

        await Promise.all(
          _.range(y.shape[1]).map(async (label) => {
            const indices_in_class = (
              await tf.whereAsync(groundTruth.equal(label))
            ).squeeze();
            const num_total = indices_in_class.shape[0];
            const tp =
              num_total > 0
                ? groundTruth
                    .gather(indices_in_class)
                    .equal(prediction.gather(indices_in_class))
                    .sum()
                    .dataSync()[0]
                : 0;
            const fn = num_total - tp;
            const recall = num_total > 0 ? tp / num_total : 0;

            const indices_predicted_in_class = (
              await tf.whereAsync(prediction.equal(label))
            ).squeeze();
            const fp =
              indices_predicted_in_class.shape[0] > 0
                ? groundTruth
                    .gather(indices_predicted_in_class)
                    .notEqual(prediction.gather(indices_predicted_in_class))
                    .sum()
                    .dataSync()[0]
                : 0;

            const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
            const f1Score = (2 * tp) / (2 * tp + fp + fn);

            setF1Scores((labels) =>
              labels.map((scores, index) =>
                index === label ? [...scores, f1Score] : scores,
              ),
            );

            updateLog(`${receivedIdx2Label[label]} (${num_total}) -------`);
            updateLog(`TP ${tp} FP ${fp} FN ${fn}`);
            updateLog(
              `F1 ${f1Score.toFixed(2)} R ${recall.toFixed(
                2,
              )} P ${precision.toFixed(2)}`,
            );
            updateLog(' ');
          }),
        );
      }

      updateLog('Training done.');

      // Free GPU memory.
      X.dispose();
      y.dispose();

      // Calc model diff.
      updateLog('Creating diff...');
      const modelDiff = await model.createSerializedDiff(modelParams);
      updateLog('Done.');

      // Report diff.
      updateLog('Reporting diff...');
      await job.report(modelDiff);
      updateLog('Done.');
    });

    job.on('rejected', ({ timeout }) => {
      // Handle the job rejection.
      if (timeout) {
        const msUntilRetry = timeout * 1000;
        // Try to join the job again in "msUntilRetry" milliseconds
        updateLog(`Rejected from cycle, retry in ${timeout}`);
        setTimeout(job.start.bind(job), msUntilRetry);
      } else {
        updateLog(
          `Rejected from cycle with no timeout, assuming Model training is complete.`,
        );
      }
    });

    job.on('error', (err) => {
      updateLog(`Error: ${err.message}`);
    });
  };

  return (
    <div className="App">
      <h1>Federated NER JS Worker</h1>
      <div style={{ display: 'flex', flexDirection: 'row' }}>
        <div>
          <div>
            <h2>Config</h2>
            <pre>
              <code>{JSON.stringify(config, null, 2)}</code>
            </pre>
          </div>
          <div>
            <h2>Setup</h2>
            <p>
              <label>Grid URL</label>
              <input
                name="text"
                value={gridUrl}
                onChange={(e) => setGridUrl(e.target.value)}
              />
            </p>
            <p>
              <label>Model Name</label>
              <input
                name="text"
                value={modelName}
                onChange={(e) => setModelName(e.target.value)}
              />
            </p>
            <p>
              <label>Model Version</label>
              <input
                name="text"
                value={modelVersion}
                onChange={(e) => setModelVersion(e.target.value)}
              />
            </p>
            <button onClick={handleSubmit}>Submit</button>
          </div>
        </div>
        <div
          style={{
            marginLeft: '30px',
            flex: 1,
            borderLeft: '1px solid hsl(200, 10%, 95%)',
            paddingLeft: '30px',
          }}
        >
          <h2>Log</h2>
          <div style={{ maxHeight: '500px', overflow: 'auto' }}>
            {log.map((entry, index) => (
              <div key={index}>
                <code>{entry}</code>
              </div>
            ))}
          </div>
        </div>
        <div
          style={{
            marginLeft: '30px',
            display: 'flex',
            flexDirection: 'column',
            borderLeft: '1px solid hsl(200, 10%, 95%)',
            flex: 1,
            paddingLeft: '30px',
          }}
        >
          <h2>Training</h2>
          <Plot
            data={[
              {
                x: _.range(losses.length),
                y: losses,
                type: 'scatter',
                mode: 'lines',
              },
            ]}
            layout={{
              yaxis: {
                rangemode: 'tozero',
                autorange: true,
              },
              xaxis: { rangemode: 'nonnegative', autorange: true },
              width: 300,
              height: 200,
              title: 'Loss',
              margin: { r: 20, l: 20, t: 30, b: 20 },
            }}
            config={{ staticPlot: true }}
          />
          <Plot
            data={[
              {
                x: _.range(accuracies.length),
                y: accuracies,
                type: 'scatter',
                mode: 'lines',
              },
            ]}
            layout={{
              xaxis: { rangemode: 'nonnegative', autorange: true },
              yaxis: { range: [0, 1] },
              width: 300,
              height: 200,
              margin: { r: 20, l: 20, t: 30, b: 20 },
              title: 'Accuracy',
            }}
            config={{ staticPlot: true }}
          />
          {f1Scores.map((scores, label) => (
            <Plot
              data={[
                {
                  x: _.range(scores.length),
                  y: scores,
                  type: 'scatter',
                  mode: 'lines',
                },
              ]}
              layout={{
                xaxis: { rangemode: 'nonnegative', autorange: true },
                yaxis: { range: [0, 1] },
                width: 300,
                height: 200,
                margin: { r: 20, l: 20, t: 30, b: 20 },
                title: `${label}: ${label in idx2Label && idx2Label[label]}`,
              }}
              config={{ staticPlot: true }}
            />
          ))}
        </div>
      </div>
    </div>
  );
};

export default App;
