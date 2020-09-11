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

export const idx2Label = ['O', 'MISC', 'ORG', 'PER', 'LOC'];
export const label2Idx = { O: 0, MISC: 1, ORG: 2, PER: 3, LOC: 4 };
const embeddingsDim = 100;

export class ConLLData {
  async loadEmbeddings() {
    const embeddingsResponse = await fetch(
      `http://localhost:8000/glove.6B.${embeddingsDim}d.txt`,
    );
    const embeddingsText = await embeddingsResponse.text();
    const embeddings = {};

    console.log('Creating embedding dictionary...');

    embeddingsText.split('\n').forEach((line) => {
      if (line.length === 0) return;
      let [token, ...embedding] = line.split(' ');
      embedding = embedding.map((num) => Number(num));
      embeddings[token.toLowerCase()] = embedding;
    });

    embeddings['UNKNOWN_TOKEN'] = _.times(embeddingsDim, _.constant(0));

    this.embeddings = embeddings;
  }

  async loadTrainDataset() {
    return await this.loadDataset('http://localhost:8000/eng.train.txt');
  }

  async loadTestDataset() {
    return await this.loadDataset('http://localhost:8000/eng.testa.txt');
  }

  async loadDataset(url) {
    console.log(`Loading dataset from ${url}.`);
    const dataResponse = await fetch(url);
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
    sentences = sentences.map((sentence) =>
      sentence.filter(([token, label]) => label !== 'O'),
    );

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
    
    let count = 0;

    console.log('Building tensors...');

    const X = tf.tensor2d(
      flatTokens.map((token) => {
        if (token in this.embeddings) {
          count++;
          return this.embeddings[token];
        }
        return this.embeddings['UNKNOWN_TOKEN'];
      }),
      [flatTokens.length, embeddingsDim],
    );
    console.log(
      `Found embeddings for ${count} of ${flatTokens.length} tokens.`,
    );
    const y = tf.tensor2d(flatLabelsOneHot, [flatTokens.length, 5]);

    return [X, y];
  }
}
