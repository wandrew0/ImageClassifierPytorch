import './App.css';

import React, {useState, useEffect} from "react";
import {InferenceSession, Tensor, env} from "onnxruntime-web";
import {Image} from "image-js";

import { classOf } from './QuickDrawClasses';

env.wasm.wasmPaths = "/static/"

const softmax = (logits) => {
  const maxLogit = Math.max(...logits);
  const exps = logits.map((x) => Math.exp(x - maxLogit));
  const sumExps = exps.reduce((a, b) => a + b);
  return exps.map((x) => x / sumExps);
};

const getTopK = (arr, k) => {
  arr = Array.from(arr);
  const idxs = arr.map((value, index) => [value, index]).sort((a, b) => b[0] - a[0]).slice(0, k).map((item) => item[1]);
  return idxs;
}

function App() {
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(null); // 0: no, 1: image, 2: classifying
  const [session, setSession] = useState(null);
  const [tensor, setTensor] = useState(null);
  const topK = 5;

  useEffect(() => {
    setLoading(-3);
    (async () => {
      const session = await InferenceSession.create("/efficientnet_v2_s_quickdraw.onnx", {graphOptimizationLevel: "all"});
      setSession(session);
    })();
  }, [])

  const handleImageChange = async(e) => {
    setLoading(-1)

    const image = URL.createObjectURL(e.target.files[0])
    setImage(image);
    const img = await Image.load(image)
    const data = img.getRGBAData({format: "Float32Array"});

    const tensor = new Float32Array(28 * 28);

    for(let i = 0; i < 28 * 28; i++) {
      tensor[i] = data[i * 4] / 255.0;
    }

    const input = new Tensor("float32", tensor, [1, 1, 28, 28]);
    setTensor(input)
    setLoading(0)
  };

  const classifyImage = async() => {
    setLoading(-2);

    const feeds = {"input": tensor};

    const start = performance.now()
    const output = await session.run(feeds);
    const end = performance.now();

    const logits = output[Object.keys(output)[0]].data;
    const probs = softmax(logits);

    const topKIndices = getTopK(probs, topK);

    const resultData = topKIndices.map((idx) => ({
      class: classOf(idx),
      prob: (probs[idx] * 100).toFixed(2)
    }));
    setResult({time: (end - start).toFixed(2), resultData});
    setLoading(0);
  };

  const loadingText = (loading) => {
    switch(loading) {
      case -3:
        return "No image.";
      case 0:
        return "Classify Image";
      case -1:
        return "Loading Image";
      case -2:
        return "Classifying...";
    }
  }

  return (
  <div className="App">
    <h1>Image Classifier</h1>
    <input type="file" accept="image/*" onChange={handleImageChange} className="file-input" />
    {image && <img src={image} style={{imageRendering:"pixelated"}} alt="Selected" className="selected-image" width="300px"/>}
    <div className="button-container">
      <button onClick={classifyImage} disabled={!image || loading != 0} className="classify-button">
        {loadingText(loading)}
      </button>
    </div>
    {result && (
        <div className="result">
          <p className="time">Time: {result.time} ms</p>
          <table className="result-table">
            <thead>
              <tr>
                <th>Class</th>
                <th>Probability</th>
              </tr>
            </thead>
            <tbody>
              {result.resultData.map((item, index) => (
                <tr key={index}>
                  <td>{item.class}</td>
                  <td>{item.prob} %</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
  </div>
  );
}

export default App;