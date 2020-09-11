/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import _ from 'lodash';
import * as tf from '@tensorflow/tfjs-core';

export class ConLLData {
  async load() {
    const dataResponse = await fetch('http://localhost:8000/eng.train.txt');
    const text = await dataResponse.text();

    let sentences = [[]];
    text.split('\n').forEach((line) => {
      if (line.includes('-DOCSTART')) return;

      if (line.length === 0) {
        sentences.push([]);
        return;
      }

      const [token, , , label] = line.split(' ');
      sentences[sentences.length - 1].push([
        token,
        label.replace('I-', '').replace('B-', ''),
      ]);
    });
    sentences = sentences.filter((sentence) => sentence.length > 0);

    const words = new Set();
    const labels = new Set();

    console.log('Extracting words and labels...');
    sentences.forEach((sentence) =>
      sentence.forEach(([token, label]) => {
        labels.add(label);
        words.add(token.toLowerCase());
      }),
    );
    console.log(`Extracted ${words.size} words and ${labels.size} labels.`);
    console.log(labels);
    /*
    const word2Idx = {};
    word2Idx['PADDING_TOKEN'] = 0;
    word2Idx['UNKNOWN_TOKEN'] = 1;

    words.forEach((word) => {
      word2Idx[word] = Object.keys(word2Idx).length;
    });

    const idx2Word = {};
    Object.keys(idx2Word).forEach((word) => {
      idx2Word[idx2Word[word]] = word;
    });
    */

    const label2Idx = {};
    labels.forEach((label) => {
      label2Idx[label] = Object.keys(label2Idx).length;
    });

    const idx2Label = {};
    Object.keys(label2Idx).forEach((label) => {
      idx2Label[label2Idx[label]] = label;
    });

    /*
    const data = sentences.map((sentence) =>
      sentence.map(([token, label]) => [
        word2Idx[token.toLowerCase()] || word2Idx['UNKNOWN_TOKEN'],
        label2Idx[label],
      ]),
    );
    */
    const embeddingsResponse = await fetch(
      'http://localhost:8000/glove.6B.50d.txt',
    );
    const embeddingsText = await embeddingsResponse.text();
    const embeddings = {};
    embeddings['UNKNOWN_TOKEN'] = _.times(50, _.constant(0));
    embeddingsText.split('\n').forEach((line) => {
      let [token, ...embedding] = line.split(' ');
      embedding = embedding.map((num) => Number(num));
      embeddings[token.toLowerCase()] = embedding;
    });

    /*
    const data = sentences.map((sentence) => {
      const tokens = sentence.map((arr) => arr[0].toLowerCase());
      const labels = sentence.map((arr) => arr[0]);

      return [
        tf.tensor2d(
          tokens.map((token) =>
            token in embeddings
              ? embeddings[token]
              : embeddings['UNKNOWN_TOKEN'],
          ),
          [tokens.length, 50],
        ),
        tf.tensor1d(labels.map((label) => label2Idx[label])),
      ];
    });
    */

    const flatTokens = sentences
      .map((sentence) => sentence.map((arr) => arr[0].toLowerCase()))
      .flat();
    const flatLabels = sentences
      .map((sentence) => sentence.map((arr) => label2Idx[arr[1]]))
      .flat();
    const flatLabelsOneHot = flatLabels.map((label) => {
      const oneHot = _.times(5, _.constant(0));
      oneHot[label] = 1;
      return oneHot;
    });

    const X = tf.tensor2d(
      flatTokens.map((token) =>
        token in embeddings ? embeddings[token] : embeddings['UNKNOWN_TOKEN'],
      ),
      [flatTokens.length, 50],
    );
    const y = tf.tensor2d(flatLabelsOneHot, [flatTokens.length, 5]);

    return { X, y, label2Idx, idx2Label };
  }

  getTrainData() {}

  getTestData(numExamples) {}
}
